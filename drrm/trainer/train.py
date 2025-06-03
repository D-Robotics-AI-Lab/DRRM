import copy
import logging
import os
from pathlib import Path
import hydra

import diffusers
import torch
from torch.utils.data import RandomSampler, BatchSampler
import transformers
from transformers.models.deit.image_processing_deit import valid_images
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from safetensors.torch import load_model

from drrm.models.ema_model import EMAModel


if is_wandb_available():
    import wandb

@torch.no_grad()
def log_sample_res(policy_model, args, dataloader, logger):
    logger.info(f"Running sampling for {args.num_val_batches} batches...")

    policy_model.eval()
    
    loss_for_log = {}
    val_losses = list()
    for step, batch in enumerate(dataloader):
        if step >= args.num_val_batches:
            break
        
        loss = policy_model(batch)
        val_losses.append(loss.item())
        
    if len(val_losses) > 0:
        val_loss = torch.mean(torch.tensor(val_losses)).item()
    
    loss_for_log['loss'] = val_loss
    
    policy_model.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)

def train(args, logger):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=Path(args.output_dir, args.logging_dir),
        project_config=ProjectConfiguration(total_limit=args.checkpoints_total_limit),
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
    val_dataset = train_dataset.get_validation_dataset()
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset) * args.num_train_epochs,
    )
    batch_sampler = BatchSampler(
        sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    max_iters = len(train_dataloader) / accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * max_iters,
        num_training_steps=max_iters,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler                   
    )

    ema_policy_model.to(accelerator.device, dtype=weight_dtype)                                                                             


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("RoboticsManipulation", config=dict(args))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num iters = {args.num_train_epochs * len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_iters / args.gradient_accumulation_steps}")
    global_step = 0

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
                accelerator.load_state(os.path.join(args.output_dir, path), strict=False)   # TODO: strict=Fasle取消掉
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                policy_model.module.load_state_dict(checkpoint["module"])
                
            load_model(ema_policy_model, os.path.join(args.output_dir, path, "ema", "model.safetensors"), strict=False)
            global_step = int(path.split("-")[1])

            # normalizer is not load to device by default, so we need to do it manually
            if hasattr(policy_model, "normalizer"):
                policy_model.normalizer.to(accelerator.device)


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, max_iters), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    loss_for_log = {}
    policy_model.train()
        
    # Forward and backward...
    for batch in train_dataloader:
        with accelerator.accumulate(policy_model):
            loss = policy_model(batch)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
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

            if args.val_period > 0 and global_step % args.val_period == 0:
                logger.info(f"Sampling at step {global_step}")
                sample_loss_for_log = log_sample_res(
                    policy_model,    # We do not use EMA currently
                    args,
                    val_dataloader,
                    logger,
                )
                logger.info(sample_loss_for_log)
                accelerator.log(sample_loss_for_log, step=global_step)
        
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        logs.update(loss_for_log)
        # logger.info(logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_iters:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(policy_model).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_policy_model, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")

    accelerator.end_training()