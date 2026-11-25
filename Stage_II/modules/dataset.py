"""
Dataset module for loading COCO format data for SAM training.
"""
import logging
from pathlib import Path
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from .config import Config


logger = logging.getLogger(__name__)


def get_transforms(image_size: int = 1024, is_train: bool = True) -> A.Compose:
    """
    Get augmentation transforms.

    Args:
        image_size: Target image size
        is_train: Whether this is for training (applies augmentations)

    Returns:
        Albumentations compose object
    """
    if is_train:
        transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    return transforms


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable number of objects per image.
    """
    # Since each image can have different number of masks, we return a list
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'masks': [item['masks'] for item in batch],
        'boxes': [item.get('boxes', None) for item in batch],
        'point_coords': [item.get('point_coords', None) for item in batch],
        'point_labels': [item.get('point_labels', None) for item in batch],
        'category_ids': [item['category_ids'] for item in batch],
        'image_ids': torch.stack([item['image_id'] for item in batch]),
        'image_names': [item['image_name'] for item in batch]
    }


class GlaucomaDataset(Dataset):
    """
    Glaucoma dataset loader for SAM training.

    Dataset structure:
        - Images: {id}_{left|right}.jpg
        - Masks: {id}_{left|right}_cup.npy, {id}_{left|right}_disc.npy
        - Split: split.json with 'train', 'test', 'validation' keys

    Classes:
        - Class 0: optic cup
        - Class 1: optic disc
    """

    def __init__(
        self,
        split_file: Path,
        dataset_root: Path,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        image_size: int = 1024,
        use_point_prompts: bool = True,
        use_box_prompts: bool = True,
        num_points: int = 5,
        mask_format: str = "npy",
        mask_suffixes: List[str] = None,
        use_only_labeled: bool = True
    ) -> None:
        """
        Args:
            split_file: Path to split.json file
            dataset_root: Root directory containing images and masks
            split: One of 'train', 'valid', 'test'
            transforms: Albumentations transforms
            image_size: Target image size
            use_point_prompts: Whether to generate point prompts
            use_box_prompts: Whether to generate box prompts
            num_points: Number of point prompts per mask
            mask_format: 'npy' or 'png'
            mask_suffixes: List of mask types to load (e.g., ['cup', 'disc'])
            use_only_labeled: If True, only use images with masks
        """
        self.dataset_root = Path(dataset_root)
        self.split_file = split_file
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        self.use_point_prompts = use_point_prompts
        self.use_box_prompts = use_box_prompts
        self.num_points = num_points
        self.mask_format = mask_format
        self.mask_suffixes = mask_suffixes or ["cup", "disc"]
        self.use_only_labeled = use_only_labeled

        # Load split file
        with open(split_file, 'r', encoding='utf-8') as f:
            self.split_data = json.load(f)

        # Map split names
        split_key = split
        if split == "valid":
            split_key = "validation"

        if split_key not in self.split_data:
            raise ValueError(f"Split '{split_key}' not found in {split_file}. Available: {list(self.split_data.keys())}")

        # Get all items for this split
        all_items = self.split_data[split_key]

        # Filter items
        self.items = []
        skipped = 0
        for item in all_items:
            image_filename = item['image_filename']
            has_masks = self._has_masks(image_filename)

            if use_only_labeled and not has_masks:
                skipped += 1
                continue

            self.items.append({
                'id': item.get('id', 0),
                'image_filename': image_filename,
                'label': item.get('label', 'Unknown'),
                'has_masks': has_masks
            })

        logger.info(f"GlaucomaDataset [{split}]: loaded {len(self.items)} images")
        if use_only_labeled:
            logger.info(f"  Skipped {skipped} images without masks")
        logger.info(f"  Mask types: {self.mask_suffixes}")
        class_names = {
            'cup': 'optic_cup',
            'disc': 'optic_disc'
        }
        classes_str = ", ".join(
            f"{i}={class_names.get(suffix, suffix)}"
            for i, suffix in enumerate(self.mask_suffixes)
        )
        logger.info(f"  Classes: {classes_str}")

    def _has_masks(self, image_filename: str) -> bool:
        """Check if all mask files exist for an image"""
        base = image_filename.replace('.jpg', '')
        for suffix in self.mask_suffixes:
            mask_path = self.dataset_root / f"{base}_{suffix}.{self.mask_format}"
            if not mask_path.is_file():
                return False
        return True

    def _load_mask(self, image_filename: str, suffix: str) -> np.ndarray:
        """Load mask for an image"""
        base = image_filename.replace('.jpg', '')
        mask_path = self.dataset_root / f"{base}_{suffix}.{self.mask_format}"

        if self.mask_format == "npy":
            mask = np.load(mask_path)
        else:
            mask = np.array(Image.open(mask_path))
            # Convert 255 to 1
            mask = (mask > 0).astype(np.uint8)

        return mask

    def _get_box_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Get bounding box from binary mask"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)

        ymin, ymax = pos[0].min(), pos[0].max()
        xmin, xmax = pos[1].min(), pos[1].max()
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    def _get_points_from_mask(
        self,
        mask: np.ndarray,
        num_points: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from mask as prompts"""
        h, w = mask.shape

        # Positive points
        pos_coords = np.argwhere(mask > 0)
        if len(pos_coords) == 0:
            pos_points = np.array([[w//2, h//2]], dtype=np.float32)
        else:
            if len(pos_coords) < num_points:
                indices = np.random.choice(len(pos_coords), num_points, replace=True)
            else:
                indices = np.random.choice(len(pos_coords), num_points, replace=False)
            pos_points = pos_coords[indices].astype(np.float32)
            pos_points = pos_points[:, [1, 0]]  # (y, x) -> (x, y)

        # Negative points
        neg_coords = np.argwhere(mask == 0)
        if len(neg_coords) > 0:
            if len(neg_coords) < num_points:
                indices = np.random.choice(len(neg_coords), num_points, replace=True)
            else:
                indices = np.random.choice(len(neg_coords), num_points, replace=False)
            neg_points = neg_coords[indices].astype(np.float32)
            neg_points = neg_points[:, [1, 0]]  # (y, x) -> (x, y)
        else:
            neg_points = np.array([], dtype=np.float32).reshape(0, 2)

        return pos_points, neg_points

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        image_filename = item['image_filename']

        # Load image
        img_path = self.dataset_root / image_filename
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Collect masks
        masks = []
        boxes = []
        point_coords_list = []
        point_labels_list = []
        category_ids = []

        if item['has_masks']:
            for class_id, suffix in enumerate(self.mask_suffixes):
                mask = self._load_mask(image_filename, suffix)

                # Ensure binary and same shape
                mask = (mask > 0).astype(np.uint8)

                if mask.sum() == 0:
                    continue

                masks.append(mask)

                # Box prompt
                if self.use_box_prompts:
                    box = self._get_box_from_mask(mask)
                    boxes.append(box)

                # Point prompts
                if self.use_point_prompts:
                    pos_points, neg_points = self._get_points_from_mask(mask, self.num_points)
                    all_points = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
                    labels = np.concatenate([
                        np.ones(len(pos_points)),
                        np.zeros(len(neg_points)) if len(neg_points) > 0 else np.array([])
                    ])
                    point_coords_list.append(all_points)
                    point_labels_list.append(labels)

                category_ids.append(class_id)

        if len(masks) == 0:
            # Dummy data for images without masks (normal images)
            # Use empty mask with a negative point prompt to indicate no object
            masks = [np.zeros((orig_h, orig_w), dtype=np.uint8)]
            boxes = [np.array([0, 0, orig_w, orig_h], dtype=np.float32)]
            point_coords_list = [np.array([[orig_w//2, orig_h//2]], dtype=np.float32)]
            point_labels_list = [np.array([0], dtype=np.float32)]  # 0 = negative label
            category_ids = [0]

        # Stack masks
        masks = np.stack(masks, axis=0)  # [N, H, W]

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                masks=[masks[i] for i in range(len(masks))]
            )
            image = transformed['image']
            masks = np.stack(transformed['masks'], axis=0) if len(transformed['masks']) > 0 else masks
        else:
            # Default resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            resized_masks = []
            for m in masks:
                resized_m = cv2.resize(
                    m.astype(np.uint8),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_masks.append(resized_m)
            masks = np.stack(resized_masks, axis=0)

            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Scale boxes and points
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        if len(point_coords_list) > 0:
            for i in range(len(point_coords_list)):
                point_coords_list[i] = point_coords_list[i].astype(np.float32)
                point_coords_list[i][:, 0] *= scale_x
                point_coords_list[i][:, 1] *= scale_y

        result = {
            'image': image,
            'masks': torch.from_numpy(masks).float(),
            'category_ids': torch.tensor(category_ids, dtype=torch.long),
            'image_id': torch.tensor(item['id'], dtype=torch.long),
            'image_name': image_filename
        }

        if self.use_box_prompts and len(boxes) > 0:
            result['boxes'] = torch.from_numpy(np.array(boxes)).float()

        if self.use_point_prompts and len(point_coords_list) > 0:
            max_points = max(len(pts) for pts in point_coords_list)
            padded_coords = []
            padded_labels = []
            for coords, labels in zip(point_coords_list, point_labels_list):
                pad_len = max_points - len(coords)
                if pad_len > 0:
                    coords = np.vstack([coords, np.zeros((pad_len, 2))])
                    labels = np.concatenate([labels, np.zeros(pad_len)])
                padded_coords.append(coords)
                padded_labels.append(labels)

            result['point_coords'] = torch.from_numpy(np.array(padded_coords)).float()
            result['point_labels'] = torch.from_numpy(np.array(padded_labels)).long()

        return result


def create_glaucoma_dataloaders(
    config: Config,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for Glaucoma dataset.

    Args:
        config: Configuration object
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    gcfg = config.glaucoma_data

    # Training dataset
    train_dataset = GlaucomaDataset(
        split_file=gcfg.get_split_file_path(),
        dataset_root=gcfg.dataset_root,
        split="train",
        transforms=get_transforms(gcfg.image_size, is_train=True),
        image_size=gcfg.image_size,
        use_point_prompts=True,
        use_box_prompts=True,
        mask_format=gcfg.mask_format,
        mask_suffixes=gcfg.mask_suffixes,
        use_only_labeled=gcfg.use_only_labeled
    )

    # Validation dataset
    val_dataset = GlaucomaDataset(
        split_file=gcfg.get_split_file_path(),
        dataset_root=gcfg.dataset_root,
        split="valid",
        transforms=get_transforms(gcfg.image_size, is_train=False),
        image_size=gcfg.image_size,
        use_point_prompts=True,
        use_box_prompts=True,
        mask_format=gcfg.mask_format,
        mask_suffixes=gcfg.mask_suffixes,
        use_only_labeled=gcfg.use_only_labeled
    )

    # Test dataset
    test_dataset = GlaucomaDataset(
        split_file=gcfg.get_split_file_path(),
        dataset_root=gcfg.dataset_root,
        split="test",
        transforms=get_transforms(gcfg.image_size, is_train=False),
        image_size=gcfg.image_size,
        use_point_prompts=True,
        use_box_prompts=True,
        mask_format=gcfg.mask_format,
        mask_suffixes=gcfg.mask_suffixes,
        use_only_labeled=gcfg.use_only_labeled
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
