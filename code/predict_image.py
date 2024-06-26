import torch
from torchvision import transforms
from PIL import Image
import os
from model import CNNVariant1

# Saved model
model_path = "Variant1_best_model.pth"  # Best model
num_classes = 4


transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((48, 48)),  # Ensure the image is 48x48
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNVariant1(num_classes=num_classes)  # Load the best model variant
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# Function to predict the class of an image
def predict_image(image_path):

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # Map the prediction to the corresponding class
    class_names = [
        "angry",
        "focused",
        "happy",
        "neutral",
    ]
    predicted_class = class_names[predicted.item()]

    return predicted_class


# Path to the image
image_path = "./Final_Images/Age/Senior/Focused/27967.jpg"

# Predict the class of the image
predicted_class = predict_image(image_path)
print(f"The predicted class for the image is: {predicted_class}")
