import torch
import torch.optim as optim
from model import ImageToFormulaModel
from utils.train_utils import train_step
from scripts.preprocessing.data_loader import get_data_loaders

input_channels = 3
hidden_size = 512
vocab_size = 5000
batch_size = 32
epochs = 10
learning_rate = 0.001
teacher_forcing_ratio = 0.5

model = ImageToFormulaModel(input_channels, hidden_size, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader, val_loader = get_data_loaders(batch_size)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        targets = targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        loss = train_step(model, images, targets, optimizer, teacher_forcing_ratio)
        running_loss += loss
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")
    
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
