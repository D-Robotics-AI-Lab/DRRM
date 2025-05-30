# 1 Introduction
记录和解释目前数据配置的方法
- self.num_epochs
```bash
# step1: 在config 对应的yaml中进行配置
# step2: 配置数据集
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
if hasattr(args, 'num_train_epochs') and args.num_train_epochs is not None:
    train_samples = args.num_train_epochs * len(train_dataset)
else:
    train_samples = args.max_train_steps

sampler = RandomSampler(
    train_dataset,
    replacement=args.train_sampler.sampler.replacement,
    num_samples=train_samples,
)
# step3: 更新学习率相关的配置
# Scheduler and math around the number of training steps.
override_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if hasattr(args, 'num_train_epochs') or args.num_train_epochs is None:
    max_train_steps = num_update_steps_per_epoch # 默认使用1个epoch
    override_max_train_steps = True
else:
    max_train_steps = args.max_train_steps
lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)
# step4: 重新配置训练epochs
# 现在epoch数量始终是1
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if override_max_train_steps:
    max_train_steps = num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# step5: checkpoint resume
resume_global_step = global_step * args.gradient_accumulation_steps
first_epoch = global_step // num_update_steps_per_epoch
resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
```
