import copy
import logging
import math
import os
from pathlib import Path
import hydra
from tqdm.auto import tqdm
import omegaconf
import typing
import collections

import torch
from torch.utils.data import RandomSampler, BatchSampler
from torch.serialization import add_safe_globals
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_model

from .sample_dp import log_sample_res


def train(args, logger):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accelerator.gradient_accumulation_steps,
        mixed_precision=args.accelerator.mixed_precision,
        log_with=args.accelerator.report_to,
        project_dir=Path(args.output_dir, args.logging_dir),
        project_config=ProjectConfiguration(total_limit=args.accelerator.checkpoints_total_limit),
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
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

    # Get weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create policy model
    policy_model = hydra.utils.instantiate(args.model)
    policy_model.to(accelerator.device, dtype=weight_dtype)

    # Create EMA model
    ema_policy_model = copy.deepcopy(policy_model)
    ema_model = hydra.utils.instantiate(args.ema, model=ema_policy_model)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(policy_model))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer creation
    optimizer = hydra.utils.instantiate(args.optimizer, params=policy_model.parameters())

    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    # Get normalizer
    if hasattr(train_dataset, "get_normalizer"):
        normalizer = train_dataset.get_normalizer()
        policy_model.set_normalizer(normalizer)
        ema_policy_model.set_normalizer(normalizer)
    # Get validation dataset from train dataset
    val_dataset = train_dataset.get_validation_dataset()
    # Set sampler and batch sampler
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset) * args.num_epochs,
    )
    batch_sampler = BatchSampler(
        sampler,
        batch_size=args.train_dataloader.batch_size,
        drop_last=args.train_dataloader.drop_last,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.train_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_dataloader.batch_size,
        shuffle=args.val_dataloader.shuffle,
        num_workers=args.val_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # compute max_train_steps
    max_train_steps = len(train_dataloader)

    lr_scheduler = get_scheduler(
        args.scheduler.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.scheduler.lr_warmup_steps * max_train_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.scheduler.lr_num_cycles,
        power=args.scheduler.lr_power,
    )

    # Prepare everything with our `accelerator`.
    policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, val_dataloader, lr_scheduler                   
    )

    ema_policy_model.to(accelerator.device, dtype=weight_dtype)                                                                             

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("RoboticsManipulation")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_dataloader.batch_size}")
    logger.info(f"  Num train steps (w. len(train_dataset), num_epochs & train_batch_size) = {max_train_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_dataloader.batch_size * args.accelerator.gradient_accumulation_steps * accelerator.num_processes}")
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
                accelerator.load_state(os.path.join(args.output_dir, path)) # strict=False
            except:
                # TODO: find a good way to load the state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                load_model(policy_model, os.path.join(args.output_dir, path, "model.safetensors"), strict=False)
                
            load_model(ema_policy_model, os.path.join(args.output_dir, path, "ema", "model.safetensors"), strict=False)
            global_step = int(path.split("-")[1])

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

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

            if args.sample_period > 0 and global_step % args.sample_period == 0:
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

        if global_step >= max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(policy_model).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_model, ema_save_path)
        
        logger.info(f"Saved Model to {args.output_dir}")

    accelerator.end_training()