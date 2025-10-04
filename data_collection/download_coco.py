#!/usr/bin/env python3
"""
COCO Dataset Downloader for BlindSpot
Downloads and filters COCO 2017 dataset for indoor obstacle classes
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Set
import requests
from tqdm import tqdm
from pycocotools.coco import COCO
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# COCO 2017 URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'train_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# Target classes for indoor obstacle detection
TARGET_CLASSES = {
    0: 'person',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'keyboard',
    66: 'cell phone',
    67: 'microwave',
    69: 'oven',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
}


class COCODownloader:
    """Download and filter COCO dataset"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / 'images'
        self.annotations_dir = self.output_dir / 'annotations'
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        self.stats = {
            'total_images': 0,
            'filtered_images': 0,
            'class_distribution': {},
            'images_per_class': {}
        }

    def download_file(self, url: str, output_path: Path, description: str):
        """Download file with progress bar"""
        if output_path.exists():
            logger.info(f"{description} already exists at {output_path}")
            return

        logger.info(f"Downloading {description}...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=description
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"✓ Downloaded {description}")

    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file"""
        import zipfile

        logger.info(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"✓ Extracted to {extract_to}")

    def download_annotations(self):
        """Download COCO annotations"""
        annotations_zip = self.output_dir / 'annotations_trainval2017.zip'

        self.download_file(
            COCO_URLS['train_annotations'],
            annotations_zip,
            'COCO Annotations'
        )

        if not (self.annotations_dir / 'instances_train2017.json').exists():
            self.extract_zip(annotations_zip, self.output_dir)

    def filter_images(self, split: str = 'train2017') -> Set[int]:
        """
        Filter images containing target classes

        Args:
            split: 'train2017' or 'val2017'

        Returns:
            Set of image IDs to download
        """
        annotation_file = self.annotations_dir / f'instances_{split}.json'

        if not annotation_file.exists():
            logger.error(f"Annotation file not found: {annotation_file}")
            return set()

        logger.info(f"Loading {split} annotations...")
        coco = COCO(str(annotation_file))

        # Get all images containing target classes
        target_image_ids = set()
        target_cat_ids = list(TARGET_CLASSES.keys())

        logger.info(f"Filtering images with target classes...")
        for cat_id in tqdm(target_cat_ids, desc="Processing classes"):
            if cat_id not in coco.cats:
                logger.warning(f"Category {cat_id} not found in COCO dataset")
                continue

            cat_name = coco.cats[cat_id]['name']
            img_ids = coco.getImgIds(catIds=[cat_id])
            target_image_ids.update(img_ids)

            # Statistics
            self.stats['images_per_class'][cat_name] = len(img_ids)
            logger.info(f"  {cat_name} (ID {cat_id}): {len(img_ids)} images")

        self.stats['total_images'] = len(coco.imgs)
        self.stats['filtered_images'] = len(target_image_ids)

        logger.info(f"\n{'='*60}")
        logger.info(f"Filtered {len(target_image_ids)} / {len(coco.imgs)} images "
                   f"({len(target_image_ids)/len(coco.imgs)*100:.1f}%)")
        logger.info(f"{'='*60}\n")

        return target_image_ids

    def download_images(self, split: str = 'train2017', image_ids: Set[int] = None):
        """
        Download filtered images

        Args:
            split: 'train2017' or 'val2017'
            image_ids: Set of image IDs to download (None = download all)
        """
        annotation_file = self.annotations_dir / f'instances_{split}.json'
        coco = COCO(str(annotation_file))

        split_dir = self.images_dir / split
        split_dir.mkdir(exist_ok=True)

        if image_ids is None:
            image_ids = set(coco.imgs.keys())

        logger.info(f"Downloading {len(image_ids)} images for {split}...")

        downloaded = 0
        skipped = 0
        failed = 0

        for img_id in tqdm(image_ids, desc=f"Downloading {split}"):
            img_info = coco.imgs[img_id]
            img_url = img_info['coco_url']
            img_filename = img_info['file_name']
            img_path = split_dir / img_filename

            # Skip if already exists
            if img_path.exists():
                skipped += 1
                continue

            try:
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()

                with open(img_path, 'wb') as f:
                    f.write(response.content)

                downloaded += 1

            except Exception as e:
                logger.error(f"Failed to download {img_filename}: {e}")
                failed += 1

        logger.info(f"\n{split} Download Summary:")
        logger.info(f"  Downloaded: {downloaded}")
        logger.info(f"  Skipped (already exist): {skipped}")
        logger.info(f"  Failed: {failed}")

    def create_filtered_annotations(self, split: str, image_ids: Set[int]):
        """
        Create filtered annotation file with only target images and classes

        Args:
            split: 'train2017' or 'val2017'
            image_ids: Set of image IDs to include
        """
        annotation_file = self.annotations_dir / f'instances_{split}.json'
        filtered_file = self.annotations_dir / f'instances_{split}_filtered.json'

        logger.info(f"Creating filtered annotations for {split}...")

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Filter images
        filtered_images = [
            img for img in coco_data['images']
            if img['id'] in image_ids
        ]

        # Filter annotations
        target_cat_ids = set(TARGET_CLASSES.keys())
        filtered_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] in image_ids and ann['category_id'] in target_cat_ids
        ]

        # Filter categories
        filtered_categories = [
            cat for cat in coco_data['categories']
            if cat['id'] in target_cat_ids
        ]

        # Create new annotation file
        filtered_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': filtered_categories
        }

        with open(filtered_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        logger.info(f"✓ Filtered annotations saved to {filtered_file}")
        logger.info(f"  Images: {len(filtered_images)}")
        logger.info(f"  Annotations: {len(filtered_annotations)}")
        logger.info(f"  Categories: {len(filtered_categories)}")

    def generate_statistics(self):
        """Generate dataset statistics"""
        stats_file = self.output_dir / 'coco_statistics.json'

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("COCO Dataset Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Total images in dataset: {self.stats['total_images']}")
        logger.info(f"Filtered images: {self.stats['filtered_images']}")
        logger.info(f"\nImages per class:")

        for class_name, count in sorted(
            self.stats['images_per_class'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {class_name}: {count}")

        logger.info(f"\n✓ Statistics saved to {stats_file}")

    def run(self, splits: List[str] = ['train2017', 'val2017'], download_images_flag: bool = True):
        """
        Run complete download pipeline

        Args:
            splits: List of splits to download ['train2017', 'val2017']
            download_images_flag: Whether to download images (False = annotations only)
        """
        logger.info("="*60)
        logger.info("COCO Dataset Download for BlindSpot")
        logger.info("="*60)

        # Step 1: Download annotations
        self.download_annotations()

        # Step 2: For each split
        for split in splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {split}")
            logger.info(f"{'='*60}\n")

            # Filter images
            image_ids = self.filter_images(split)

            if not image_ids:
                logger.warning(f"No images found for {split}, skipping...")
                continue

            # Create filtered annotations
            self.create_filtered_annotations(split, image_ids)

            # Download images (optional - can be large!)
            if download_images_flag:
                self.download_images(split, image_ids)
            else:
                logger.info(f"Skipping image download for {split} (--no-images flag)")

        # Step 3: Generate statistics
        self.generate_statistics()

        logger.info("\n" + "="*60)
        logger.info("✓ COCO Download Complete!")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Download and filter COCO dataset for BlindSpot'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/coco',
        help='Output directory for COCO data'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train2017', 'val2017'],
        choices=['train2017', 'val2017'],
        help='Splits to download'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Download annotations only (skip images - saves bandwidth)'
    )

    args = parser.parse_args()

    # Create downloader and run
    downloader = COCODownloader(args.output_dir)
    downloader.run(
        splits=args.splits,
        download_images_flag=not args.no_images
    )


if __name__ == '__main__':
    main()
