import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(512*16*16, hidden_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        attn_weights = torch.matmul(
            torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))),
            self.v.unsqueeze(1)
        ).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights

class ImageToFormulaModel(nn.Module):
    def __init__(self, input_channels, hidden_size, vocab_size, num_layers=1):
        super(ImageToFormulaModel, self).__init__()
        self.encoder = Encoder(input_channels, hidden_size, hidden_size)
        self.attention = Attention(hidden_size, hidden_size)
        self.decoder = Decoder(hidden_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        encoder_outputs = self.encoder(images).unsqueeze(1)  # [batch_size, 1, hidden_size]
        decoder_input = torch.zeros(batch_size, 1, encoder_outputs.size(-1)).to(images.device)
        hidden = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.hidden_size).to(images.device)

        outputs = []
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        decoder_input = decoder_input + context
        
        for t in range(targets.size(1)):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)

            top1 = output.argmax(2)
            decoder_input = top1.unsqueeze(1).float()  # Teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t].unsqueeze(1).float()
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
