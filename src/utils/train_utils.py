import torch
import torch.nn.functional as F

def train_step(model, images, targets, optimizer, teacher_forcing_ratio=0.5):
    model.train()
    optimizer.zero_grad()
    outputs = model(images, targets, teacher_forcing_ratio)
    
    # Flatten outputs and targets
    outputs = outputs.view(-1, outputs.size(-1))
    targets = targets.view(-1)
    
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    
    return loss.item()
