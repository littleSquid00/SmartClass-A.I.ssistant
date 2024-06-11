import sys
sys.path.append('./models')
from data_loading import load_all_data
# from models.model1 import Model1
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model):
    # Get Data
    train_loader, _ = load_all_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), 'model_info/model1/model1.pth')
    print("Training Complete")

def evaluate_model(model):
    _ , test_loader = load_all_data()
    # Example: Iterate through the test dataset
    model.load_state_dict(torch.load('model_info/model1/model1.pth'))
    model.eval()
    test_inputs, test_labels = next(iter(test_loader))
    test_outputs = model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    correct_predictions = torch.sum(predicted == test_labels).item()
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Total number of test samples: {len(test_labels)}")
    print(f"Accuracy: {correct_predictions / len(test_labels) * 100:.2f}%")

    f = open('model_info/model1/performance.txt', 'w')
    f.write(f'Test Accuracy: {correct_predictions / len(test_labels) * 100:.2f}%')
    f.close()
