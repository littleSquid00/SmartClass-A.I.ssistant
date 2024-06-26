import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import CNNVariant2
import os

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Datasets
root_image_folder = "Final_Images"
age_folder = os.path.join(root_image_folder, "Age")
gender_folder = os.path.join(root_image_folder, "Gender")

age_datasets = {}
gender_datasets = {}

## Load Age datasets
for age_subfolder in os.listdir(age_folder):
    subfolder_path = os.path.join(age_folder, age_subfolder)
    if os.path.isdir(subfolder_path):
        age_datasets[age_subfolder] = datasets.ImageFolder(root=subfolder_path, transform=transform)

## Load Gender datasets
for gender_subfolder in os.listdir(gender_folder):
    subfolder_path = os.path.join(gender_folder, gender_subfolder)
    if os.path.isdir(subfolder_path):
        gender_datasets[gender_subfolder] = datasets.ImageFolder(root=subfolder_path, transform=transform)

## Create Data Loaders
age_loaders = {}
gender_loaders = {}
for age_group, dataset in age_datasets.items():
    age_loaders[age_group] = DataLoader(dataset, batch_size=25, shuffle=False)
for gender, dataset in gender_datasets.items():
    gender_loaders[gender] = DataLoader(dataset, batch_size=25, shuffle=False)

# Load the Best Model from Part II
model = CNNVariant2(num_classes=num_classes)

# Keep one of the following 2 lines commented and the other uncommented
model.load_state_dict(torch.load(f"Variant2_best_model.pth")) # Model V2
# model.load_state_dict(torch.load(f"Variant2.1_best_model.pth")) # Model V2.1

model.to(device)
model.eval()


# Evaluation function
def evaluate_model(model, data_loader, category_name, class_name):
    y_true = []
    y_pred = []

    # Iterate over the data loader
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification report
    print(f"Classification Report for Model on {category_name} - {class_name}:")
    print(classification_report(y_true, y_pred, target_names=data_loader.dataset.classes, digits=4))


# Evaluate model on each Age and Gender class
for age_class, loader in age_loaders.items():
    evaluate_model(model, loader, "Age", age_class)

for gender_class, loader in gender_loaders.items():
    evaluate_model(model, loader, "Gender", gender_class)
