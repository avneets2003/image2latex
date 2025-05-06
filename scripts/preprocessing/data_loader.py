import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.utils.latex_utils import tokenize_latex

class LaTeXDataset(Dataset):
    def __init__(self, image_list_path, image_dir, vocab, transform=None):
        self.image_list_path = image_list_path
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.image_paths, self.formulas = self.load_data()

    def load_data(self):
        image_paths = []
        formulas = []
        with open(self.image_list_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                img_path, formula = line.strip().split()
                image_paths.append(os.path.join(self.image_dir, img_path))
                formulas.append(formula)
        return image_paths, formulas

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        formula = self.formulas[idx]
        image = Image.open(img_path).convert('RGB')  # Load image in RGB format

        if self.transform:
            image = self.transform(image)

        # Tokenize the LaTeX formula
        tokens = tokenize_latex(formula)
        # Convert tokens to indices based on vocab
        indices = [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]

        return image, torch.tensor(indices)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def create_dataloader(image_list_path, image_dir, vocab, batch_size=32):
    dataset = LaTeXDataset(image_list_path, image_dir, vocab, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
