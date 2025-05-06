from PIL import Image
import os
import torchvision.transforms as transforms

def load_image(image_path):
    """Load an image from the specified path."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, save_path):
    """Save an image to the specified path."""
    try:
        image.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")

def image_transform(image, size=(128, 128)):
    """Apply basic transformations to an image."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)
