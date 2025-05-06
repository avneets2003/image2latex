import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.model.model import Im2LatexModel
from scripts.preprocessing.data_loader import load_vocab

def evaluate(model_path, image_dir, image_list_file, vocab_path, device='cuda'):
    # Load vocab
    _, idx2token = load_vocab(vocab_path)

    # Load model
    model = Im2LatexModel(len(idx2token))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Image pre-processing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((160, 500)),
        transforms.ToTensor()
    ])

    with open(image_list_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        image_name = line.strip().split()[0]
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tokens = model.predict(image_tensor)

        # Convert output to LaTeX
        output_latex = ' '.join([idx2token[t] for t in output_tokens if t in idx2token])
        print(f"{image_name}: {output_latex}")
