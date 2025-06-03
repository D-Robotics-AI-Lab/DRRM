# based on https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/train/train.py
import copy
import logging
import math
import os
from pathlib import Path
import hydra

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from safetensors.torch import load_model

from drrm.models.ema_model import EMAModel
from .sample import log_sample_res

if is_wandb_available():
    import wandb

# logger 是经过accelerate.logging.get_logger包装的logger
def train(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit) # 限制checkpoint数量: 40
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,  # TODO:尝试各种精度的训练时间
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log the accelerator state information on all processes for debugging
    # This includes information about distributed training setup, device allocation,
    # mixed precision settings, and other accelerator configuration details
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Policy Model creation
    policy_model = hydra.utils.instantiate(args.model)
    policy_model.to(accelerator.device, dtype=weight_dtype)

    ema_policy_model = copy.deepcopy(policy_model)
    ema_model = EMAModel(
        ema_policy_model,
        update_after_step=args.ema.update_after_step,
        inv_gamma=args.ema.inv_gamma,
        power=args.ema.power,
        min_value=args.ema.min_value,
        max_value=args.ema.max_value,
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(policy_model))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO: 
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    #TODO: delete this learning rate scaling
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer creation
    params_to_optimize = policy_model.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    if hasattr(train_dataset, "get_normalizer"):
        normalizer = train_dataset.get_normalizer()
        policy_model.set_normalizer(normalizer)
        ema_policy_model.set_normalizer(normalizer)
    if hasattr(train_dataset, "get_validation_dataset"):
        eval_dataset = train_dataset.get_validation_dataset()
    else:
        eval_dataset = hydra.utils.instantiate(args.eval_dataset)

    if hasattr(args, "total_batch_size") and args.total_batch_size is not None:
        args.train_batch_size = args.total_batch_size // accelerator.num_processes
        args.sample_batch_size = args.total_batch_size // accelerator.num_processes
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    if hasattr(args, "train_sampler"):
        from torch.utils.data import RandomSampler, BatchSampler
        # TODO:兼容到num_epochs
        # Calculate total training steps for the sampler
        if hasattr(args, 'num_train_epochs') and args.num_train_epochs is not None:
            train_samples = args.num_train_epochs * len(train_dataset)
        else:
            train_samples = args.max_train_steps
        
        sampler = RandomSampler(
            train_dataset,
            replacement=args.train_sampler.sampler.replacement,
            num_samples=train_samples,
        )
        batch_sampler = BatchSampler(
            sampler,
            batch_size=args.train_batch_size,
            drop_last=args.train_sampler.drop_last,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
    else:    
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
    # TODO: sample_dataloader 的名称
    sample_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.sample_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch
    # TODO: 设置warmup 比例，推荐值5%
    if hasattr(args, 'lr_warmup_ratio') and args.lr_warmup_ratio is not None:
        lr_warmup_steps = int(args.lr_warmup_ratio * max_train_steps)
    else:
        lr_warmup_steps = args.lr_warmup_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    policy_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler                   
    )

    ema_policy_model.to(accelerator.device, dtype=weight_dtype)                                                                             


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("RoboticsManipulation", config=dict(args))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                policy_model.module.load_state_dict(checkpoint["module"])
                
            load_model(ema_policy_model, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_for_log = {}
    for epoch in range(first_epoch, num_train_epochs):

        policy_model.train()
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for batch in train_dataloader:
            with accelerator.accumulate(policy_model):
                loss = policy_model(batch)

                accelerator.backward(loss)

                # 打印一些本身参数有梯度，但是backward发现没有梯度的参数
                # for name, param in policy_model.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print("未获得梯度的参数:", name, param.shape)
                if accelerator.sync_gradients:
                    # TODO: 梯度裁剪
                    params_to_clip = policy_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            ema_model.step(accelerator.unwrap_model(policy_model))

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_policy_model, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    logger.info(f"Sampling at step {global_step}")
                    sample_loss_for_log = log_sample_res(
                        policy_model,    # We do not use EMA currently
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            # logger.info(logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(policy_model).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_policy_model, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")

    accelerator.end_training()