from PIL import Image
import os
import torchvision.transforms as transforms

def preprocess_image(image_path, size=(128, 128)):
    """Load and preprocess the image."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

if __name__ == "__main__":
    image_dir = "data/sample/images"
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            image = preprocess_image(img_path)
