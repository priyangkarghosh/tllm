import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.nn import functional as F
from model.config import GPTConfig
from model.model import GPT


def train(
    device: torch.device,
    model_config: GPTConfig,
    train_loader,
    num_epochs: int,
    learning_rate: float = 3e-4,
    checkpoints_path: str = './checkpoints',
) -> GPT:
    os.makedirs(checkpoints_path, exist_ok=True)

    model = GPT(model_config).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(device.type)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch in loop:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.6f} | LR: {lr:.6e}")

        # optional checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f"epoch_{epoch+1}.pt"))
    return model