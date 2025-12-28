import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


class UNetDataset(Dataset):
    """
    U-Net dataset with optional border-weight map computation
    following the original U-Net paper.
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        transform=None, # This transform will be DoubleCompose (e.g., augs + ToTensor)
        compute_weight_map=True,
        w0=10,
        sigma=5,
        crop_output_size=None
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.compute_weight_map = compute_weight_map
        self.w0 = w0
        self.sigma = sigma
        self.crop_output_size = crop_output_size

        print(f"DEBUG: UNetDataset initialized with crop_output_size: {self.crop_output_size}")

        self.images = sorted([f for f in os.listdir(image_dir)
                            if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        self.labels = sorted([f for f in os.listdir(label_dir)
                           if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])

        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) and labels ({len(self.labels)}) must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label (NumPy arrays)
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = io.imread(img_path)
        label = io.imread(label_path)

        # Ensure (H, W)
        if image.ndim == 3:
            image = image[..., 0]
        if label.ndim == 3:
            label = label[..., 0]

        # Normalize image to [0, 1] - applies to original image size
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        # Binary label (0, 1)
        label = (label > 0).astype(np.uint8)

        weight_map = None
        if self.compute_weight_map:
            weight_map = self._compute_weight_map(label) # Computes on original label size

        # --- IMPORTANT: Apply cropping to label and weight_map *BEFORE* transforms ---
        # Image is NOT cropped here; it's fed to UNet at original size.
        # Labels and weight_map MUST match the UNet's *output* size.
        if self.crop_output_size is not None:
            h_orig, w_orig = label.shape
            target_h, target_w = self.crop_output_size, self.crop_output_size

            if h_orig > target_h or w_orig > target_w:
                start_h = (h_orig - target_h) // 2
                start_w = (w_orig - target_w) // 2
                label = label[start_h:start_h + target_h, start_w:start_w + target_w]
                if weight_map is not None:
                    weight_map = weight_map[start_h:start_h + target_h, start_w:start_w + target_w]
                print(f"DEBUG: Label shape AFTER cropping (NumPy): {label.shape}") # Debug print
            elif h_orig < target_h or w_orig < target_w:
                # This case should ideally not happen if input images are 512x512 and target is 324x324
                raise ValueError(f"Cropped label size {h_orig}x{w_orig} is smaller than target crop size {target_h}x{target_w}. Check model architecture or input sizes.")


        # Apply transforms (e.g., augmentation + ToTensor). These now operate on
        # (original size) image and (cropped) label/weight_map.
        print(f"DEBUG IN UNetDataset (Before transforms): image.shape={image.shape}, label.shape={label.shape}") # Debug print
        if self.transform:
            if weight_map is None:
                image, label = self.transform(image, label) # image is 512x512, label is now 324x324
            else:
                image, label, weight_map = self.transform(image, label, weight_map) # image is 512x512, label, weight_map are now 324x324

        # Final checks for tensor types and dimensions (mostly handled by DoubleToTensor if it's the last transform).
        # Ensure label is long and 2D
        if label.ndim == 3: # If somehow DoubleToTensor added a channel dimension to label
            label = label.squeeze(0)

        # Ensure weight_map is float and 2D
        if weight_map is not None:
            if weight_map.ndim == 3: # If somehow DoubleToTensor added a channel dimension to weight_map
                weight_map = weight_map.squeeze(0)

        # Ensure correct tensor types (DoubleToTensor should have done this, but a safety check)
        image = image.float()
        label = label.long()

        print(f"DEBUG IN UNetDataset (After final type conversion): label.shape = {label.shape}") # Debug print

        if weight_map is not None:
            weight_map = weight_map.float()
            return {
                "image": image,
                "label": label,
                "weight_map": weight_map
            }

        return {
            "image": image,
            "label": label
        }

    def _compute_weight_map(self, label_mask):
        """
        Compute pixel-wise weight map that emphasizes borders
        between touching objects (Equation 2 in U-Net paper).
        """
        label_mask = label_mask.astype(np.uint8)

        # Label connected components
        labeled_mask = label(label_mask)
        num_objects = labeled_mask.max()

        # If 0 or 1 objects, return uniform weights
        if num_objects <= 1:
            return np.ones(label_mask.shape, dtype=np.float32)

        # Compute distance to each object
        distances = np.zeros((label_mask.shape[0], label_mask.shape[1], num_objects), dtype=np.float32)

        for i in range(1, num_objects + 1):
            obj = (labeled_mask == i).astype(np.uint8)
            distances[:, :, i - 1] = distance_transform_edt(1 - obj)

        # Find smallest and second smallest distances
        distances_sorted = np.sort(distances, axis=2)
        d1 = distances_sorted[:, :, 0]  # Distance to nearest object
        d2 = distances_sorted[:, :, 1]  # Distance to second nearest object

        # Border weight (higher at borders between objects)
        border_weight = self.w0 * np.exp(-((d1 + d2) ** 2) / (2 * self.sigma ** 2))

        # Class balancing weight
        n_pixels = label_mask.size
        n_fg = np.sum(label_mask == 1)
        n_bg = np.sum(label_mask == 0)

        wc = np.ones(label_mask.shape, dtype=np.float32)
        if n_fg > 0 and n_bg > 0:
            wc[label_mask == 1] = n_bg / n_fg
            wc[label_mask == 0] = 1.0

        weight_map = wc + border_weight
        return weight_map
