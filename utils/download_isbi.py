"""
Download and setup ISBI 2012 Cell Segmentation Dataset
Run this in Google Colab to automatically download the dataset
"""

import os
import urllib.request
import zipfile
from PIL import Image
import numpy as np

PROJECT_ROOT = "/content/drive/MyDrive/unet-project"

def download_isbi_2012():
    """
    Download ISBI 2012 Cell Segmentation Challenge dataset
    """
    print("="*60)
    print("Downloading ISBI 2012 Cell Segmentation Dataset")
    print("="*60)

    # Create directories
    data_dir = f"{PROJECT_ROOT}/data"
    os.makedirs(data_dir, exist_ok=True)

    # URLs for the dataset (these are example URLs - you may need to update)
    # The official dataset is available at: http://brainiac2.mit.edu/isbi_challenge/

    print("\n⚠️  IMPORTANT: Download the dataset manually")
    print("="*60)
    print("1. Go to: http://brainiac2.mit.edu/isbi_challenge/")
    print("2. Download:")
    print("   - train-volume.tif (training images)")
    print("   - train-labels.tif (training labels)")
    print("3. Upload to your Google Drive")
    print("="*60)

    # Check if files exist
    train_volume_path = f"{data_dir}/train-volume.tif"
    train_labels_path = f"{data_dir}/train-labels.tif"

    if os.path.exists(train_volume_path) and os.path.exists(train_labels_path):
        print("\n✅ Dataset files found!")
        print("Processing dataset...")
        process_isbi_dataset(train_volume_path, train_labels_path)
    else:
        print("\n❌ Dataset files not found")
        print(f"Please upload files to: {data_dir}/")
        print("   - train-volume.tif")
        print("   - train-labels.tif")


def process_isbi_dataset(volume_path, labels_path):
    """
    Process ISBI 2012 dataset: extract slices from TIF stacks
    """
    print("\nProcessing dataset...")

    # Create output directories
    images_dir = f"{PROJECT_ROOT}/data/train-volume/images"
    labels_dir = f"{PROJECT_ROOT}/data/train-labels/labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load TIF stacks
    print("Loading training volume...")
    volume = Image.open(volume_path)

    print("Loading training labels...")
    labels = Image.open(labels_path)

    # Extract and save individual slices
    n_frames = volume.n_frames
    print(f"Found {n_frames} slices")

    for i in range(n_frames):
        # Extract image slice
        volume.seek(i)
        img = np.array(volume)

        # Extract label slice
        labels.seek(i)
        label = np.array(labels)

        # Save as PNG
        img_filename = f"slice_{i:03d}.png"
        Image.fromarray(img).save(f"{images_dir}/{img_filename}")
        Image.fromarray(label).save(f"{labels_dir}/{img_filename}")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_frames} slices")

    print(f"\n✅ Dataset processed successfully!")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")
    print(f"Total: {n_frames} image-label pairs")

    # Display sample
    print("\n" + "="*60)
    print("Sample images:")
    print("="*60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx in range(3):
        # Load image and label
        img_path = f"{images_dir}/slice_{idx:03d}.png"
        label_path = f"{labels_dir}/slice_{idx:03d}.png"

        img = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))

        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(f'Image {idx}')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(label, cmap='gray')
        axes[1, idx].set_title(f'Label {idx}')
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/data/sample_images.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Sample visualization saved to: {PROJECT_ROOT}/data/sample_images.png")


def verify_dataset():
    """
    Verify that the dataset is properly set up
    """
    images_dir = f"{PROJECT_ROOT}/data/train-volume/images"
    labels_dir = f"{PROJECT_ROOT}/data/train-labels/labels"

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Dataset not found!")
        return False

    images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])

    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    if len(images) != len(labels):
        print("❌ Mismatch between images and labels!")
        return False

    # Check image sizes
    if len(images) > 0:
        img = Image.open(f"{images_dir}/{images[0]}")
        label = Image.open(f"{labels_dir}/{labels[0]}")
        print(f"Image size: {img.size}")
        print(f"Label size: {label.size}")

        if img.size != label.size:
            print("❌ Image and label sizes don't match!")
            return False

    print("✅ Dataset is properly set up!")
    return True


# Main execution
if __name__ == "__main__":
    download_isbi_2012()

    # If dataset is processed, verify it
    if os.path.exists(f"{PROJECT_ROOT}/data/train-volume/images"):
        verify_dataset()

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Verify your config.yaml has:")
    print("   images_dir: 'data/train-volume/images'")
    print("   labels_dir: 'data/train-labels/labels'")
    print("2. Run: python train.py")
    print("3. Run: python evaluate.py")
