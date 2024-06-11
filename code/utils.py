import sys
sys.path.append('./models')
from data_loading import load_all_data
# from models.model1 import Model1
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, model_name):
    # Get Data
    train_loader, val_loader, _ = load_all_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Initialize variables to track the best model and early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

         # Check if the current model is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience and epoch < 9:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Save the best model
    torch.save(best_model_state, f'model_info/{model_name}/{model_name}.pth')
    print('Best model saved with validation loss:', best_val_loss)

    # Save Performance
    f = open(f'model_info/{model_name}/valid_performance.txt', 'w')
    f.write(f'Validation Loss: {best_val_loss:.2f}, Validation Accuracy: {best_val_acc*100:.2f}%')
    f.close()

def evaluate_model(model, model_name):
    _ , _ , test_loader = load_all_data()
    # Load Best Model
    model.load_state_dict(torch.load(f'model_info/{model_name}/{model_name}.pth'))
    model.eval()
    test_inputs, test_labels = next(iter(test_loader))
    test_outputs = model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    correct_predictions = torch.sum(predicted == test_labels).item()
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Total number of test samples: {len(test_labels)}")
    print(f"Accuracy: {correct_predictions / len(test_labels) * 100:.2f}%")

    f = open(f'model_info/{model_name}/test_performance.txt', 'w')
    f.write(f'Test Accuracy: {correct_predictions / len(test_labels) * 100:.2f}%')
    f.close()
