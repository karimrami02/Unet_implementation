import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from models.unet import UNet
from data.dataset import UNetDataset
from metrics.metrics import MetricTracker
from utils.data_utils import get_train_val_split, SubsetDataset
from utils.transforms import DoubleCompose, DoubleToTensor


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_predictions(images, labels, predictions, save_dir, num_samples=5):
    """Visualize and save prediction results"""
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        # Images from dataloader might be normalized, denormalize for display if needed
        img = images[i].squeeze().cpu().numpy()
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Ground truth
        label = labels[i].cpu().numpy()
        axes[1].imshow(label, cmap='gray')
        axes[1].set_title('Ground Truth (Cropped)') # Indicate it's cropped
        axes[1].axis('off')

        # Prediction
        pred = predictions[i].cpu().numpy()
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()


def save_predictions(predictions, save_dir, filenames):
    """Save prediction masks as images"""
    os.makedirs(save_dir, exist_ok=True)

    for pred, filename in zip(predictions, filenames):
        pred_np = pred.cpu().numpy().astype(np.uint8) * 255
        save_path = os.path.join(save_dir, filename)
        io.imsave(save_path, pred_np)


def evaluate(model, dataloader, device, metric_tracker):
    """Evaluate model on test set"""
    model.eval()
    metric_tracker.reset()

    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating")

        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Update metrics
            metric_tracker.update(predictions, labels)

            # Store for visualization
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predictions.cpu())

            # Update progress bar
            metrics = metric_tracker.get_metrics()
            loop.set_postfix(
                dice=metrics['dice'],
                iou=metrics['iou'],
                acc=metrics['accuracy']
            )

    metrics = metric_tracker.get_metrics()

    return metrics, all_images, all_labels, all_predictions


def main():
    # Load configuration
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)

    # Setup device
    device = torch.device(config['hardware']['device']
                         if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build transforms (no augmentation for evaluation, just ToTensor)
    test_transform = DoubleCompose([DoubleToTensor()])

    # Define the expected output size from UNet for 512x512 input
    unet_output_size = 324 # Corrected output size

    # Check if separate test set exists
    if config['data']['test_images'] and config['data']['test_labels']:
        print("Using separate test set")
        test_dataset = UNetDataset(
            image_dir=os.path.join(project_root, config['data']['test_images']),
            label_dir=os.path.join(project_root, config['data']['test_labels']),
            transform=test_transform,
            compute_weight_map=False,
            crop_output_size=unet_output_size # Pass the output size here
        )
    else:
        print("No separate test set found. Using validation split.")
        # Use validation split from training data
        image_dir = os.path.join(project_root, config['data']['images_dir'])
        label_dir = os.path.join(project_root, config['data']['labels_dir'])

        # Get the same split as training (needed to recreate the split)
        train_files, val_files = get_train_val_split(
            image_dir=image_dir,
            label_dir=label_dir,
            train_ratio=config['data']['train_val_split'],
            random_seed=config['data']['random_seed']
        )

        # Build full dataset (only for validation files, so it can be passed to SubsetDataset)
        # This dataset will use the test_transform for processing.
        full_val_dataset_for_eval = UNetDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=test_transform, # Apply test transform directly
            compute_weight_map=False, # Not needed for evaluation
            crop_output_size=unet_output_size # Pass the output size here
        )

        # Use validation subset
        test_dataset = SubsetDataset(full_val_dataset_for_eval, val_files)

    print(f"Test samples: {len(test_dataset)}")

    # Build dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    # Build model
    model = UNet(
        n_channels=config['model']['n_channels']
        , n_classes=config['model']['n_classes']
    )
    model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(
        project_root,
        config['checkpoint']['save_dir'],
        'unet_best.pth'
    )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Looking for latest checkpoint...")
        checkpoint_dir = os.path.join(project_root, config['checkpoint']['save_dir'])
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoints found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Metric tracker
    metric_tracker = MetricTracker()

    # Evaluate
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")

    metrics, images, labels, predictions = evaluate(
        model, test_loader, device, metric_tracker
    )

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Dice Score : {metrics['dice']:.4f}")
    print(f"IoU Score  : {metrics['iou']:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print("="*50 + "\n")

    # Save visualizations
    if config['evaluation']['save_predictions']:
        pred_dir = os.path.join(project_root, config['evaluation']['prediction_dir'])

        print("Saving visualizations...")
        visualize_predictions(images, labels, predictions, pred_dir, num_samples=10)
        print(f"Visualizations saved to: {pred_dir}")

        print("Saving prediction masks...")
        mask_dir = os.path.join(pred_dir, 'masks')
        filenames = [f"pred_{i+1:03d}.png" for i in range(len(predictions))]
        save_predictions(predictions, mask_dir, filenames)
        print(f"Prediction masks saved to: {mask_dir}")

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
