"""
Train/Val/Test Split Generator
Split YOLO format dataset into training, validation, and test sets
"""

import os
import shutil
from pathlib import Path
from typing import Tuple
import random
from tqdm import tqdm


class DatasetSplitter:
    """Split YOLO format dataset into train/val/test"""

    def __init__(self, yolo_dir: str, output_dir: str = None):
        """
        Args:
            yolo_dir: Directory containing YOLO format dataset
            output_dir: Output directory for splits (default: same as yolo_dir)
        """
        self.yolo_dir = Path(yolo_dir)
        self.output_dir = Path(output_dir) if output_dir else self.yolo_dir

        # Source directories
        self.src_images_dir = self.yolo_dir / 'images' / 'train'
        self.src_labels_dir = self.yolo_dir / 'labels' / 'train'

        if not self.src_images_dir.exists():
            raise ValueError(f"Images directory not found: {self.src_images_dir}")
        if not self.src_labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.src_labels_dir}")

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05, seed: int = 42):
        """
        Split dataset into train/val/test sets

        Args:
            train_ratio: Ratio of training data (default 0.8 = 80%)
            val_ratio: Ratio of validation data (default 0.15 = 15%)
            test_ratio: Ratio of test data (default 0.05 = 5%)
            seed: Random seed for reproducibility
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Get all label files
        label_files = sorted(self.src_labels_dir.glob('*.txt'))
        total_samples = len(label_files)

        print(f"Total samples: {total_samples}")
        print(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")

        # Shuffle with fixed seed for reproducibility
        random.seed(seed)
        random.shuffle(label_files)

        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size

        # Split file lists
        train_files = label_files[:train_size]
        val_files = label_files[train_size:train_size + val_size]
        test_files = label_files[train_size + val_size:]

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_files)} samples ({len(train_files)/total_samples:.1%})")
        print(f"  Val:   {len(val_files)} samples ({len(val_files)/total_samples:.1%})")
        print(f"  Test:  {len(test_files)} samples ({len(test_files)/total_samples:.1%})")

        # Create split directories and move files
        self._create_split('train', train_files)
        self._create_split('val', val_files)
        self._create_split('test', test_files)

        print(f"\n✓ Dataset split complete!")
        print(f"  Output: {self.output_dir}")

        # Update dataset.yaml
        self._update_dataset_yaml()

    def _create_split(self, split_name: str, label_files: list):
        """Create a split by moving files to new directories"""
        print(f"\nCreating {split_name} split...")

        # Create directories
        images_dir = self.output_dir / 'images' / split_name
        labels_dir = self.output_dir / 'labels' / split_name

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Move/copy files
        for label_file in tqdm(label_files, desc=f"Processing {split_name}"):
            # Get corresponding image file
            image_name = label_file.stem + '.jpg'
            src_image = self.src_images_dir / image_name
            src_label = label_file

            # Destination paths
            dst_image = images_dir / image_name
            dst_label = labels_dir / label_file.name

            # Skip if source image doesn't exist
            if not src_image.exists():
                continue

            # Create symlinks (or copy if symlink fails)
            if split_name == 'train':
                # Train split: move symlinks from original location
                if src_image.is_symlink():
                    # Copy symlink target
                    try:
                        os.symlink(src_image.resolve(), dst_image)
                    except FileExistsError:
                        pass
                else:
                    # Move actual file
                    if not dst_image.exists():
                        shutil.move(str(src_image), str(dst_image))

                # Move label
                if not dst_label.exists():
                    shutil.move(str(src_label), str(dst_label))
            else:
                # Val/Test splits: create new symlinks
                if src_image.is_symlink():
                    target = src_image.resolve()
                    try:
                        os.symlink(target, dst_image)
                    except FileExistsError:
                        pass
                else:
                    if not dst_image.exists():
                        shutil.copy2(str(src_image), str(dst_image))

                # Copy label
                if not dst_label.exists():
                    shutil.copy2(str(src_label), str(dst_label))

    def _update_dataset_yaml(self):
        """Update dataset.yaml with new split paths"""
        yaml_path = self.output_dir / 'dataset.yaml'

        if not yaml_path.exists():
            print(f"Warning: dataset.yaml not found at {yaml_path}")
            return

        # Read existing yaml
        with open(yaml_path, 'r') as f:
            content = f.read()

        # Update paths
        content = content.replace('images/train', 'images/train')

        # Ensure val and test paths are correct
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('val:'):
                lines[i] = 'val: images/val'
            elif line.startswith('test:'):
                lines[i] = 'test: images/test'

        # Write updated yaml
        with open(yaml_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✓ Updated {yaml_path}")


def main():
    """Example usage"""
    yolo_dir = 'data/processed/yolo_format'

    # Create splitter
    splitter = DatasetSplitter(yolo_dir)

    # Split dataset: 80% train, 15% val, 5% test
    splitter.split(
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.05,
        seed=42
    )

    print("\n✓ Train/val/test splits created!")
    print("Dataset ready for YOLOv8 training")


if __name__ == '__main__':
    main()
