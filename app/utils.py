# app/utils.py

from PIL import Image
from torchvision import transforms

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Make sure this matches training
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])
    return transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
