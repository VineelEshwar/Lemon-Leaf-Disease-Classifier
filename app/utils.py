# app/utils.py

from PIL import Image
from torchvision import transforms

def preprocess_image(image):
    """
    Preprocesses an uploaded image for model prediction.
    Ensures it is a PIL RGB image, resizes, normalizes, and adds batch dimension.
    """
    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Convert to RGB to avoid issues with RGBA/Grayscale
    image = image.convert("RGB")

    # Transform pipeline (match training config)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match your model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    # Apply transforms and add batch dimension [1, C, H, W]
    return transform(image).unsqueeze(0)
