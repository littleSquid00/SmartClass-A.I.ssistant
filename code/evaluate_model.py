import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import CNN, CNNVariant1, CNNVariant2

# Define transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load all dataset
dataset = datasets.ImageFolder(root="./processed_data/alldataset", transform=transform)
num_classes = 4  # Number of classes in your dataset

# Create a DataLoader for the test set
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load the best models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "Main_Model": CNN(num_classes=num_classes),
    "Variant1": CNNVariant1(num_classes=num_classes),
    "Variant2": CNNVariant2(num_classes=num_classes),
}

for model_name in models:
    models[model_name].load_state_dict(torch.load(f"{model_name}_best_model.pth"))
    models[model_name].to(device)
    models[model_name].eval()


# evaluate a model
def evaluate_model(model, model_name):
    y_true = []
    y_pred = []

    # Iterate over the test data
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # classification report
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes,
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# Evaluate all models
for model_name, model in models.items():
    evaluate_model(model, model_name)
