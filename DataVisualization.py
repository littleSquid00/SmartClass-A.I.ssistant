import os
import cv2  # Import OpenCV
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the path to your dataset
dataset_path = os.path.join(os.getcwd(), 'Emotions')

# List of class names
classes = ['Angry', 'Engaged', 'Happy', 'Neutral'] # or any chosen fourth class

# Count the number of images in each class
class_counts = {class_name: len(os.listdir(os.path.join(dataset_path, class_name))) for class_name in classes}

# Plot the class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()

# Function to load images from a folder
def load_images_from_folder(folder, sample_size=None):
    images = []
    filenames = os.listdir(folder)
    if sample_size is not None:
        filenames = random.sample(filenames, sample_size)
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is not None:
            images.append(img)
    return images

# Plot histograms for pixel intensities
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, class_name in enumerate(classes):
    images = load_images_from_folder(os.path.join(dataset_path, class_name))
    all_pixels = np.concatenate([img.flatten() for img in images])
    
    axs[i // 2, i % 2].hist(all_pixels, bins=256, range=(0, 256), color='gray', alpha=0.75)
    axs[i // 2, i % 2].set_title(f'{class_name} Pixel Intensity Distribution')
    axs[i // 2, i % 2].set_xlabel('Pixel Intensity')
    axs[i // 2, i % 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Function to plot sample images and their histograms
def plot_sample_images_with_histograms(classes, dataset_path, sample_size=15):
    for class_name in classes:
        fig, axs = plt.subplots(5, 6, figsize=(20, 15))
        images = load_images_from_folder(os.path.join(dataset_path, class_name), sample_size)
        
        for j, img in enumerate(images):
            row = j // 3
            col_img = (j % 3) * 2
            col_hist = col_img + 1
            
            # Plot the image
            axs[row, col_img].imshow(img, cmap='gray')
            axs[row, col_img].axis('off')
            axs[row, col_img].set_title(f'{class_name} sample {j + 1}')
            
            # Plot the histogram
            axs[row, col_hist].hist(img.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.75)
            axs[row, col_hist].set_xlabel('Pixel Intensity')
            axs[row, col_hist].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

# Display sample images and their histograms
plot_sample_images_with_histograms(classes, dataset_path, sample_size=15)