from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

def load_all_data(mini_batch_size=50):
    # Define the transform to convert the image to a tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((48, 48)),  # Resize to 48x48 pixels
        transforms.ToTensor()  # Convert to tensor
    ])

    train_dataset = ImageFolder(root='data/train', transform=transform)
    test_dataset = ImageFolder(root='data/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader
