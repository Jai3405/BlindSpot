"""
COCO to YOLO Format Converter
Converts COCO JSON annotations to YOLO format for training YOLOv8
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm


class COCOtoYOLO:
    """Convert COCO format annotations to YOLO format"""

    def __init__(self, coco_json_path: str, images_dir: str, output_dir: str):
        """
        Args:
            coco_json_path: Path to COCO JSON annotation file
            images_dir: Directory containing COCO images
            output_dir: Output directory for YOLO format dataset
        """
        self.coco_json_path = coco_json_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        # Load COCO annotations
        print(f"Loading COCO annotations from {coco_json_path}...")
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create category ID to index mapping (YOLO uses 0-indexed classes)
        self.category_mapping = self._create_category_mapping()

        print(f"Loaded {len(self.coco_data['images'])} images")
        print(f"Found {len(self.coco_data['annotations'])} annotations")
        print(f"Classes: {len(self.category_mapping)}")

    def _create_category_mapping(self) -> Dict[int, int]:
        """Create mapping from COCO category IDs to YOLO class indices"""
        categories = sorted(self.coco_data['categories'], key=lambda x: x['id'])
        mapping = {cat['id']: idx for idx, cat in enumerate(categories)}

        # Save class names for reference
        self.class_names = [cat['name'] for cat in categories]

        return mapping

    def _convert_bbox_coco_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)

        Args:
            bbox: COCO format bbox [x, y, width, height] in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            YOLO format bbox [x_center, y_center, width, height] normalized to 0-1
        """
        x, y, w, h = bbox

        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2

        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # Clip to valid range [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        return x_center_norm, y_center_norm, w_norm, h_norm

    def convert(self, split_name: str = 'train'):
        """
        Convert COCO annotations to YOLO format

        Args:
            split_name: Name of the split (train, val, test)
        """
        # Create output directories
        images_output_dir = self.output_dir / 'images' / split_name
        labels_output_dir = self.output_dir / 'labels' / split_name

        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)

        # Create image_id to annotations mapping for faster lookup
        print("Creating image-to-annotations mapping...")
        img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Process each image
        print(f"Converting {len(self.coco_data['images'])} images to YOLO format...")
        skipped_images = 0

        for img_info in tqdm(self.coco_data['images'], desc=f"Converting {split_name}"):
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # Source image path
            src_img_path = self.images_dir / img_filename

            # Check if image exists
            if not src_img_path.exists():
                skipped_images += 1
                continue

            # Create symlink instead of copying (saves space)
            dst_img_path = images_output_dir / img_filename
            if not dst_img_path.exists():
                try:
                    os.symlink(src_img_path.absolute(), dst_img_path)
                except OSError:
                    # If symlink fails, skip (images stay in original location)
                    pass

            # Create YOLO format label file
            label_filename = Path(img_filename).stem + '.txt'
            label_path = labels_output_dir / label_filename

            # Get annotations for this image
            annotations = img_to_anns.get(img_id, [])

            # Write YOLO format annotations
            with open(label_path, 'w') as f:
                for ann in annotations:
                    category_id = ann['category_id']
                    bbox = ann['bbox']

                    # Convert to YOLO class index
                    yolo_class_idx = self.category_mapping[category_id]

                    # Convert bbox to YOLO format
                    x_center, y_center, width, height = self._convert_bbox_coco_to_yolo(
                        bbox, img_width, img_height
                    )

                    # Write YOLO annotation line: class_idx x_center y_center width height
                    f.write(f"{yolo_class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"✓ Conversion complete!")
        print(f"  Images: {len(self.coco_data['images']) - skipped_images}")
        print(f"  Skipped (missing): {skipped_images}")
        print(f"  Output: {self.output_dir}")

        # Save class names
        self._save_class_names()

        return images_output_dir, labels_output_dir

    def _save_class_names(self):
        """Save class names to a file for reference"""
        classes_file = self.output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"✓ Class names saved to {classes_file}")

    def create_dataset_yaml(self, train_path: str, val_path: str = None, test_path: str = None):
        """
        Create dataset.yaml file for YOLOv8 training

        Args:
            train_path: Path to training images directory
            val_path: Path to validation images directory (optional)
            test_path: Path to test images directory (optional)
        """
        yaml_content = f"""# BlindSpot Dataset Configuration for YOLOv8
# Indoor obstacle detection for visually impaired navigation

# Paths (relative to this file)
path: {self.output_dir.absolute()}
train: images/train
val: images/{val_path if val_path else 'val'}
test: images/{test_path if test_path else 'test'}

# Classes
nc: {len(self.class_names)}  # number of classes
names:
"""

        # Add class names
        for idx, class_name in enumerate(self.class_names):
            yaml_content += f"  {idx}: {class_name}\n"

        # Save yaml file
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"✓ Dataset YAML saved to {yaml_path}")


def main():
    """Example usage"""
    # Configuration
    coco_json = 'data/raw/coco/annotations/instances_train2017_filtered.json'
    images_dir = 'data/raw/coco/images/train2017'
    output_dir = 'data/processed/yolo_format'

    # Convert COCO to YOLO
    converter = COCOtoYOLO(coco_json, images_dir, output_dir)
    converter.convert(split_name='train')

    # Create dataset.yaml
    converter.create_dataset_yaml(train_path='train')

    print("\n✓ COCO to YOLO conversion complete!")
    print(f"Dataset ready at: {output_dir}")


if __name__ == '__main__':
    main()
