import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

mini_batch_size = 50

# Define the transform to convert the image to a tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48 pixels
    transforms.ToTensor()  # Convert to tensor
])

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # Shape: (48, 48, 16)
            nn.BatchNorm2d(16),  # Batch Normalization layer
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2), # Shape: (24, 24, 16)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # Shape: (24, 24, 32)
            nn.BatchNorm2d(32),  # Batch Normalization layer
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2) # Shape: (12, 12, 32)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=4608, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        flattened_out = torch.flatten(out, 1)
        out = self.fc_layer(flattened_out)
        out = F.log_softmax(out, dim=1)
        return out

train_dataset = ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
model = Model1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

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

torch.save(model.state_dict(), 'model_info/model1.pth')
