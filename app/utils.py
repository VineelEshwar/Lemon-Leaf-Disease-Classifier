from PIL import Image
from torchvision import transforms

def preprocess_image(image):
    """
    Preprocesses any uploaded image into a tensor ready for the model.
    Always forces RGB to avoid torchvision errors with alpha or palette modes.
    """
    # Ensure it's a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Convert to RGB no matter what
    try:
        image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not convert image to RGB: {e}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension
