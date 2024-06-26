import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, datasets

# Assuming CNNVariant2 is your model class
from model import CNNVariant2  # Adjust the import based on your actual model classes

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20
patience = 5

# Define transforms
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load your dataset
dataset = datasets.ImageFolder(root="./processed_data/alldataset", transform=transform)
num_classes = 4  # Number of classes

# Set up k-fold cross-validation
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to perform k-fold cross-validation
def evaluate_model(model_class, dataset, kfold, model_name):
    # Collect performance metrics
    performance_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    # K-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Sample elements randomly from a given list of indices
        train_subsampler = Subset(dataset, train_idx)
        test_subsampler = Subset(dataset, test_idx)

        # Further split train_subsampler into training and validation subsets
        n_train = int(len(train_subsampler) * 0.85)
        n_val = len(train_subsampler) - n_train
        train_subset, val_subset = random_split(train_subsampler, [n_train, n_val])

        # Define data loaders for training and testing
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = model_class(num_classes=num_classes)
        model.to(device)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"{model_name}, Fold {fold}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), f"{model_name}_fold{fold}_best_model.pth"
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")

        performance_metrics["accuracy"].append(accuracy)
        performance_metrics["precision"].append(precision)
        performance_metrics["recall"].append(recall)
        performance_metrics["f1_score"].append(f1)

        print(
            f"{model_name} Fold {fold} -- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )

    # Average performance metrics
    avg_accuracy = np.mean(performance_metrics["accuracy"])
    avg_precision = np.mean(performance_metrics["precision"])
    avg_recall = np.mean(performance_metrics["recall"])
    avg_f1 = np.mean(performance_metrics["f1_score"])

    print(
        f"{model_name} Average -- Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {avg_f1:.4f}"
    )

    return performance_metrics


# Evaluate both models
variant2_metrics = evaluate_model(CNNVariant2, dataset, kfold, "CNNVariant2")
variant2_1_metrics = evaluate_model(
    CNNVariant2, dataset, kfold, "CNNVariant2.1"
)  # Adjust if Variant2.1 is a different class

# Compare performance metrics
print("Comparison of Models:")
print("Metric\t\tVariant2\tVariant2.1")
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(
        f"{metric.capitalize()}\t{np.mean(variant2_metrics[metric]):.4f}\t\t{np.mean(variant2_1_metrics[metric]):.4f}"
    )
