from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    rdt, args, 
    accelerator, weight_dtype, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()
    
    loss_for_log = {}
    val_losses = list()
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break
        
        loss = rdt.compute_loss(batch)
        val_losses.append(loss.item())
        
    if len(val_losses) > 0:
        val_loss = torch.mean(torch.tensor(val_losses)).item()
    
    loss_for_log['loss'] = val_loss
    
    rdt.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)