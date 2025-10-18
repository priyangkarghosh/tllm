import os
import torch
import torch.optim as optim
from model import GPT, GPTConfig
from datetime import datetime

def save_checkpoint(
    dir: str,
    step: int,
    model: GPT,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    loss: float,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    loss_str = f"{loss:.4f}".replace('.', 'p')
    name = f"SS_L{loss_str}_{timestamp}.pt"
    path = os.path.join(dir, name)
    
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': vars(model.config),
        'loss': loss,
    }, path) 
    
def load_checkpoint(
    path: str,
    device: torch.device,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> tuple[GPT, optim.Optimizer, optim.lr_scheduler.LRScheduler, int, float]:
    checkpoint = torch.load(path, map_location=device)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint missing model config")
    
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print("[Checkpoint] Model restored")

    # load states of optimizer and scheduler
    if "optimizer" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))
    print(f"Checkpoint Resumed from step {step} | loss={loss:.4f}")
    return model, optimizer, scheduler, step, loss