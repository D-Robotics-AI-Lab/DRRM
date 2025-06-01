from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    policy_model, args, dataloader, logger
):
    logger.info(f"Running sampling for {args.num_sample_batches} batches..."
    )

    policy_model.eval()
    
    loss_for_log = {}
    val_losses = list()
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break
        
        loss = policy_model(batch)
        val_losses.append(loss.item())
        
    if len(val_losses) > 0:
        val_loss = torch.mean(torch.tensor(val_losses)).item()
    
    loss_for_log['loss'] = val_loss
    
    policy_model.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)