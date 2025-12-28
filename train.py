import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from models.unet import UNet, init_weights
from data.dataset import UNetDataset
from losses.losses import WeightedCrossEntropyLoss, DiceLoss, CombinedLoss
from metrics.metrics import MetricTracker
from utils.data_utils import get_train_val_split, SubsetDataset
from utils.transforms import (
    DoubleCompose,
    DoubleToTensor,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleVerticalFlip,
    DoubleRandomRotation,
    Normalize
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_augmentation_transforms(config):
    """Build data augmentation pipeline (Numpy-based, without ToTensor)"""
    transforms = []

    aug_config = config['augmentation']

    if aug_config['elastic_transform']['enabled']:
        transforms.append(DoubleElasticTransform(
            alpha=aug_config['elastic_transform']['alpha'],
            sigma=aug_config['elastic_transform']['sigma'],
            p=aug_config['elastic_transform']['p']
        ))

    if aug_config['horizontal_flip']['enabled']:
        transforms.append(DoubleHorizontalFlip(
            p=aug_config['horizontal_flip']['p']
        ))

    if aug_config['vertical_flip']['enabled']:
        transforms.append(DoubleVerticalFlip(
            p=aug_config['vertical_flip']['p']
        ))

    if aug_config['rotation']['enabled']:
        transforms.append(DoubleRandomRotation(
            p=aug_config['rotation']['p']
        ))

    if aug_config['normalize']['enabled']:
        transforms.append(Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        ))

    return DoubleCompose(transforms)


def build_loss_fn(config):
    """Build loss function"""
    loss_config = config['training']['loss']

    if loss_config['type'] == 'weighted_ce':
        return WeightedCrossEntropyLoss()
    elif loss_config['type'] == 'dice':
        return DiceLoss()
    elif loss_config['type'] == 'combined':
        return CombinedLoss(
            ce_weight=loss_config['ce_weight'],
            dice_weight=loss_config['dice_weight']
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")


def build_optimizer(model, config):
    """Build optimizer"""
    opt_config = config['training']

    if opt_config['optimizer'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=opt_config['learning_rate'],
            momentum=opt_config['momentum']
        )
    elif opt_config['optimizer'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=opt_config['learning_rate']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")


def build_scheduler(optimizer, config):
    """Build learning rate scheduler"""
    sched_config = config['training']

    if sched_config['scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_config['scheduler'] == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=sched_config['gamma']
        )
    elif sched_config['scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['epochs']
        )
    elif sched_config['scheduler'] is None or sched_config['scheduler'] == 'null':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_config['scheduler']}")


def validate_epoch(model, dataloader, loss_fn, device, metric_tracker):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0.0
    metric_tracker.reset()

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation")

        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)

            # Compute loss
            if 'weight_map' in batch:
                weight_map = batch['weight_map'].to(device)
                loss = loss_fn(outputs, labels, weight_map)
            else:
                loss = loss_fn(outputs, labels)

            epoch_loss += loss.item()

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            metric_tracker.update(preds, labels)

            # Update progress bar
            metrics = metric_tracker.get_metrics()
            loop.set_postfix(
                loss=loss.item(),
                dice=metrics['dice'],
                iou=metrics['iou']
            )

    epoch_loss /= len(dataloader)
    metrics = metric_tracker.get_metrics()

    return epoch_loss, metrics


def train_epoch(model, dataloader, loss_fn, optimizer, device, metric_tracker):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    metric_tracker.reset()

    loop = tqdm(dataloader, desc="Training")

    for batch in loop:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Compute loss
        if 'weight_map' in batch:
            weight_map = batch['weight_map'].to(device)
            loss = loss_fn(outputs, labels, weight_map)
        else:
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            metric_tracker.update(preds, labels)

        # Update progress bar
        metrics = metric_tracker.get_metrics()
        loop.set_postfix(
            loss=loss.item(),
            dice=metrics['dice'],
            iou=metrics['iou']
        )

    epoch_loss /= len(dataloader)
    metrics = metric_tracker.get_metrics()

    return epoch_loss, metrics


def main():
    # Load configuration
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)

    # Set random seed
    set_seed(config['seed'])

    # Setup device
    device = torch.device(config['hardware']['device']
                         if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Define the expected output size from UNet for 512x512 input -> 324x324 output
    unet_output_size = 324 # Corrected output size

    # Build augmentation transforms (NumPy-based)
    aug_transforms_list = build_augmentation_transforms(config).transforms

    # Train transform: Augmentations + ToTensor
    train_full_transform = DoubleCompose(aug_transforms_list + [DoubleToTensor()])

    # Validation transform: Only ToTensor (no augmentations)
    val_full_transform = DoubleCompose([DoubleToTensor()])

    # Get image and label directories
    image_dir = os.path.join(project_root, config['data']['images_dir'])
    label_dir = os.path.join(project_root, config['data']['labels_dir'])

    # Split into train and validation
    train_files, val_files = get_train_val_split(
        image_dir=image_dir,
        label_dir=label_dir,
        train_ratio=config['data']['train_val_split'],
        random_seed=config['data']['random_seed']
    )

    # Create train and validation UNetDataset instances
    # The dataset now handles cropping internally based on crop_output_size
    train_dataset_full = UNetDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=train_full_transform,
        compute_weight_map=config['training']['weight_map']['enabled'],
        w0=config['training']['weight_map']['w0'],
        sigma=config['training']['weight_map']['sigma'],
        crop_output_size=unet_output_size
    )
    train_dataset = SubsetDataset(train_dataset_full, train_files)


    val_dataset_full = UNetDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=val_full_transform,
        compute_weight_map=config['training']['weight_map']['enabled'],
        w0=config['training']['weight_map']['w0'],
        sigma=config['training']['weight_map']['sigma'],
        crop_output_size=unet_output_size
    )
    val_dataset = SubsetDataset(val_dataset_full, val_files)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    # Build model
    model = UNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes']
    )
    model.apply(init_weights)
    model.to(device)

    # DEBUG: Temporary test to check model output shape in train.py
    dummy_input = torch.randn(1, config['model']['n_channels'], 512, 512).to(device) # Changed dummy input to 512x512
    dummy_output = model(dummy_input)
    print(f"DEBUG: Model output shape in train.py: {dummy_output.shape}")
    del dummy_input, dummy_output # Clean up

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build loss, optimizer, scheduler
    loss_fn = build_loss_fn(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Metric tracker
    metric_tracker = MetricTracker()

    # Training loop
    best_dice = 0.0
    best_val_dice = 0.0
    start_epoch = 1

    # Resume from checkpoint if specified
    if config['checkpoint']['resume']:
        checkpoint = torch.load(config['checkpoint']['resume'])
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}")

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 50)

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, metric_tracker
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, metric_tracker
        )

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"Train IoU : {train_metrics['iou']:.4f} | Val IoU : {val_metrics['iou']:.4f}")
        print(f"Train Acc : {train_metrics['accuracy']:.4f} | Val Acc : {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        save_checkpoint = False

        if epoch % config['checkpoint']['save_freq'] == 0:
            save_checkpoint = True
            checkpoint_name = f"unet_epoch_{epoch}.pth"

        if config['checkpoint']['save_best'] and val_metrics['dice'] > best_val_dice:
            save_checkpoint = True
            checkpoint_name = "unet_best.pth"
            best_val_dice = val_metrics['dice']
            print(f"✨ New best Val Dice score: {best_val_dice:.4f}")

        if save_checkpoint:
            checkpoint_path = os.path.join(
                project_root,
                config['checkpoint']['save_dir'],
                checkpoint_name
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_val_dice': best_val_dice
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_name}")

    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best Val Dice score: {best_val_dice:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
