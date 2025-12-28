import os
import random
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def get_train_val_split(image_dir, label_dir, train_ratio=0.8, random_seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        train_ratio: Ratio of training data (default 0.8 = 80% train, 20% val)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_files, val_files: Lists of filenames for train and validation
    """
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
    
    # Verify corresponding labels exist
    label_files = [f for f in os.listdir(label_dir) 
                   if f.endswith(('.png', '.jpg', '.tif', '.tiff'))]
    
    valid_files = []
    for img_file in image_files:
        if img_file in label_files:
            valid_files.append(img_file)
        else:
            print(f"Warning: No label found for {img_file}")
    
    if len(valid_files) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    
    # Split into train and validation
    train_files, val_files = train_test_split(
        valid_files,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Train: {len(train_files)} images")
    print(f"Val:   {len(val_files)} images")
    
    return train_files, val_files


def get_kfold_splits(image_dir, label_dir, n_splits=5, random_seed=42):
    """
    Generate K-Fold cross-validation splits.
    Useful for small datasets like ISBI 2012.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        n_splits: Number of folds (default 5)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_files, val_files) tuples for each fold
    """
    # Get all valid files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
    
    label_files = [f for f in os.listdir(label_dir) 
                   if f.endswith(('.png', '.jpg', '.tif', '.tiff'))]
    
    valid_files = [f for f in image_files if f in label_files]
    
    if len(valid_files) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    print(f"Generating {n_splits}-fold cross-validation splits")
    
    # Create K-Fold splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(valid_files)):
        train_files = [valid_files[i] for i in train_idx]
        val_files = [valid_files[i] for i in val_idx]
        splits.append((train_files, val_files))
        print(f"Fold {fold_idx + 1}: Train={len(train_files)}, Val={len(val_files)}")
    
    return splits


class SubsetDataset:
    """
    Wrapper dataset that only returns specific files from the parent dataset.
    """
    
    def __init__(self, parent_dataset, file_list):
        self.parent_dataset = parent_dataset
        self.file_list = set(file_list)
        
        # Filter indices
        self.indices = []
        for idx in range(len(parent_dataset)):
            filename = parent_dataset.images[idx]
            if filename in self.file_list:
                self.indices.append(idx)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.parent_dataset[original_idx]
