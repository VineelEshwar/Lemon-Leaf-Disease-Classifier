# app/utils.py

from PIL import Image
from torchvision import transforms
import numpy as np

def preprocess_image(image):
    """
    Preprocess image for EfficientNet model.
    Converts all formats to RGB and prepares tensor for model.
    """
    # Ensure we have a PIL image object
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Force-load image into memory and convert to RGB
    image = image.convert("RGB").copy()

    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # match training size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transforms
    tensor = transform(image)

    # Add batch dimension: [1, 3, 224, 224]
    return tensor.unsqueeze(0)
