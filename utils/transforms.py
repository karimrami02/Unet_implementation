import torch
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import random


class DoubleCompose:
    """Compose multiple transforms for image and mask"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, weight_map=None):
        for t in self.transforms:
            if weight_map is not None:
                image, mask, weight_map = t(image, mask, weight_map)
            else:
                image, mask = t(image, mask)

        if weight_map is not None:
            return image, mask, weight_map
        return image, mask


class DoubleToTensor:
    """Convert numpy arrays to PyTorch tensors"""

    def __call__(self, image, mask, weight_map=None):
        # Add channel dimension if needed
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        if weight_map is not None:
            weight_map = torch.from_numpy(weight_map.copy()).float()
            # Ensure weight_map is 2D, as expected by UNetDataset
            if weight_map.ndim == 3:
                weight_map = weight_map.squeeze(0)
            return image, mask, weight_map

        return image, mask


class DoubleHorizontalFlip:
    """Randomly flip image and mask horizontally"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight_map=None):
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                if weight_map is not None:
                    weight_map = torch.flip(weight_map, dims=[-1])
            else:
                image = np.flip(image, axis=-1)
                mask = np.flip(mask, axis=-1)
                if weight_map is not None:
                    weight_map = np.flip(weight_map, axis=-1)

        if weight_map is not None:
            return image, mask, weight_map
        return image, mask


class DoubleVerticalFlip:
    """Randomly flip image and mask vertically"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight_map=None):
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
                if weight_map is not None:
                    weight_map = torch.flip(weight_map, dims=[-2])
            else:
                image = np.flip(image, axis=-2)
                mask = np.flip(mask, axis=-2)
                if weight_map is not None:
                    weight_map = np.flip(weight_map, axis=-2)

        if weight_map is not None:
            return image, mask, weight_map
        return image, mask


class DoubleRandomRotation:
    """Randomly rotate image and mask by 90, 180, or 270 degrees"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight_map=None):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            if isinstance(image, torch.Tensor):
                image = torch.rot90(image, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
                if weight_map is not None:
                    weight_map = torch.rot90(weight_map, k, dims=[-2, -1])
            else:
                image = np.rot90(image, k, axes=(-2, -1))
                mask = np.rot90(mask, k, axes=(-2, -1))
                if weight_map is not None:
                    weight_map = np.rot90(weight_map, k, axes=(-2, -1))

        if weight_map is not None:
            return image, mask, weight_map
        return image, mask


class DoubleElasticTransform:
    """
    Elastic deformation of images as described in U-Net paper.
    Applies the same deformation to both image and mask, handling different sizes.

    Args:
        alpha: Scaling factor for deformation
        sigma: Standard deviation for Gaussian filter
        p: Probability of applying the transform
    """

    def __init__(self, alpha=250, sigma=10, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask, weight_map=None):
        if random.random() > self.p:
            if weight_map is not None:
                return image, mask, weight_map
            return image, mask

        # Convert tensors to numpy if needed
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image_np = image.squeeze(0).numpy() if image.dim() == 3 else image.numpy()
            mask_np = mask.numpy()
            weight_map_np = weight_map.numpy() if weight_map is not None else None
        else:
            image_np = image
            mask_np = mask
            weight_map_np = weight_map

        # Image dimensions (e.g., 512x512)
        image_h, image_w = image_np.shape[-2:]
        # Mask/Weight map dimensions (e.g., 324x324) - these are already cropped NumPy arrays
        mask_h, mask_w = mask_np.shape[-2:]

        print(f"DEBUG IN DoubleElasticTransform (Received): image_np.shape={image_np.shape}, mask_np.shape={mask_np.shape}") # Debug print

        # 1. Generate displacement fields for the FULL image size (e.g., 512x512)
        random_state = np.random.RandomState(None) # Use a fixed random state for reproducibility if desired
        dx = gaussian_filter((random_state.rand(image_h, image_w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((random_state.rand(image_h, image_w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x_coords_img, y_coords_img = np.meshgrid(np.arange(image_w), np.arange(image_h))

        # 2. Apply deformation to the image
        indices_image = (y_coords_img + dy).reshape(-1), (x_coords_img + dx).reshape(-1)
        image_deformed = map_coordinates(image_np, indices_image, order=1, mode='reflect').reshape(image_h, image_w)

        # 3. Crop displacement fields to the mask/weight_map size (e.g., 324x324)
        # Calculate crop coordinates for deformation fields
        start_h_mask = (image_h - mask_h) // 2
        start_w_mask = (image_w - mask_w) // 2
        
        dx_cropped = dx[start_h_mask : start_h_mask + mask_h, start_w_mask : start_w_mask + mask_w]
        dy_cropped = dy[start_h_mask : start_h_mask + mask_h, start_w_mask : start_w_mask + mask_w]

        x_coords_mask, y_coords_mask = np.meshgrid(np.arange(mask_w), np.arange(mask_h))
        indices_mask = (y_coords_mask + dy_cropped).reshape(-1), (x_coords_mask + dx_cropped).reshape(-1)

        # 4. Apply deformation to the mask and weight_map using cropped fields
        mask_deformed = map_coordinates(mask_np, indices_mask, order=0, mode='reflect').reshape(mask_h, mask_w)
        
        if weight_map_np is not None:
            weight_map_deformed = map_coordinates(weight_map_np, indices_mask, order=1, mode='reflect').reshape(mask_h, mask_w)

        # Convert back to tensors if needed
        if is_tensor:
            image = torch.from_numpy(image_deformed).unsqueeze(0).float()
            mask = torch.from_numpy(mask_deformed).long()
            if weight_map is not None:
                weight_map = torch.from_numpy(weight_map_deformed).float()
                return image, mask, weight_map
            return image, mask

        print(f"DEBUG IN DoubleElasticTransform (Returning): image_deformed.shape={image_deformed.shape}, mask_deformed.shape={mask_deformed.shape}") # Debug print
        if weight_map_np is not None:
            return image_deformed, mask_deformed, weight_map_deformed
        return image_deformed, mask_deformed


class Normalize:
    """Normalize image to zero mean and unit variance"""

    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, weight_map=None):
        if isinstance(image, torch.Tensor):
            image = (image - self.mean) / self.std
        else:
            image = (image - self.mean) / self.std

        if weight_map is not None:
            return image, mask, weight_map
        return image, mask
