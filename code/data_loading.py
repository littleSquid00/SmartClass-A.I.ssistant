from torch.utils.data import DataLoader, Dataset, random_split
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
    train_size = int(0.75 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    test_dataset = ImageFolder(root='data/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader
