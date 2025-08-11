# app/utils.py

from PIL import Image
from torchvision import transforms

def preprocess_image(image):
    """
    Preprocess image for EfficientNet model.
    Works with Streamlit file uploads (jpg, png, etc.).
    Converts image to RGB, resizes, normalizes, and adds batch dimension.
    """
    # If it's a file-like object, open it
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Force RGB (avoids errors from RGBA, L, P, CMYK modes)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Define transforms (match training settings)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension
