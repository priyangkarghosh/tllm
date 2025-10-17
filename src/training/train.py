import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import GPT
from datetime import datetime
from training.data_loader import DataLoader

def train(
    device: torch.device,
    model: GPT,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    num_steps: int = 50,
    grad_accum_steps: int = 1,
    snapshot_dir: str = "./snapshots",
    snapshot_interval: int = 10,
    resume_path: str = None  # path to snapshot to resume from
):
    # create snapshot directory if it doesn't exist
    os.makedirs(snapshot_dir, exist_ok=True)
        
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
        if step % snapshot_interval == 0 or step == num_steps:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            loss_str = f"{total_loss.item():.4f}".replace(".", "p")
            name = f"ss_step{step}_loss{loss_str}_{timestamp}.pt"
            path = os.path.join(snapshot_dir, name)
            
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss.item(),
            }, path)
            
            tqdm.write(f"snapshot saved to {path}")
    
    print("Training complete")
    return model
