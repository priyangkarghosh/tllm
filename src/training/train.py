import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import GPT
from datetime import datetime
from training.data_loader_lite import DataLoaderLite

def resume_training(
    device: torch.device,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    checkpoint: str,
) -> None:
    pass

def train(
    device: torch.device,
    model: GPT,
    train_loader: DataLoaderLite,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    num_steps: int = 50,
    grad_accum_steps: int = 1,
    checkpoint: str = "./checkpoint",
    snapshot_interval: int = -1,
    resume_path: str = None  # path to snapshot to resume from
) -> None:
    # create snapshot directory if it doesn't exist
    os.makedirs(checkpoint, exist_ok=True)
        
    # resume from snapshot if specified
    start = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from snapshot: {resume_path}")
        snapshot = torch.load(resume_path, map_location=device, weights_only=False)
        
        # load state dicts from snapshot
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        scheduler.load_state_dict(snapshot['scheduler_state_dict'])
        
        start = snapshot['step'] + 1
        print(f"Resumed from step {start}")
    
    # create progress bar for training loop
    loop = tqdm(
        range(start, num_steps), initial=start, 
        total=num_steps, desc="Training"
    )
    
    for step in loop:
        optimizer.zero_grad()
        
        # gradient accumulation loop
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            logits, loss = model(x, y)
            
            # scale loss for gradient accumulation
            loss /= grad_accum_steps
            total_loss += loss.detach()
            loss.backward()
        
        # clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update learning rate and optimizer
        optimizer.step()
        scheduler.step()

        # update progress bar
        loop.set_postfix({
            'loss': f'{total_loss.item():.6f}',
            'norm': f'{norm:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # save snapshot
        step += 1
        if snapshot_interval != -1 and (step % snapshot_interval == 0 or step == num_steps):
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            loss_str = f"{total_loss.item():.4f}".replace('.', 'p')
            name = f"SS_L{loss_str}_{timestamp}.pt"
            path = os.path.join(checkpoint, name)
            
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': vars(model.config),
                'loss': total_loss.item(),
            }, path)

    print("Finished training")