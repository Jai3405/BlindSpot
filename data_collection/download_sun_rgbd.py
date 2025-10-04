#!/usr/bin/env python3
"""
SUN RGB-D Dataset Downloader for BlindSpot
Downloads and processes SUN RGB-D dataset for indoor scenes with depth information
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm
import json
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SUN RGB-D Dataset URLs
# Note: The full dataset is ~18GB compressed
SUNRGBD_URLS = {
    'data_v1': 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip',
    'toolbox': 'http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip',
    'metadata': 'http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat'
}

# Target indoor scene types
TARGET_SCENE_TYPES = {
    'bedroom',
    'bathroom',
    'living_room',
    'kitchen',
    'dining_room',
    'office',
    'home_office',
    'corridor',
    'hallway',
    'stairs',
    'furniture_store',
    'bookstore',
    'study',
    'study_room',
    'computer_room',
    'rest_space',
}


class SUNRGBDDownloader:
    """Download and process SUN RGB-D dataset"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / 'images'
        self.depth_dir = self.output_dir / 'depth'
        self.annotations_dir = self.output_dir / 'annotations'

        self.images_dir.mkdir(exist_ok=True)
        self.depth_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        self.stats = {
            'total_scenes': 0,
            'filtered_scenes': 0,
            'scene_distribution': {},
        }

    def download_file(self, url: str, output_path: Path, description: str):
        """Download file with progress bar"""
        if output_path.exists():
            logger.info(f"{description} already exists at {output_path}")
            return True

        logger.info(f"Downloading {description}...")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

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
            return True

        except Exception as e:
            logger.error(f"Failed to download {description}: {e}")
            return False

    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file"""
        import zipfile

        logger.info(f"Extracting {zip_path.name}...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total file count for progress bar
                total_files = len(zip_ref.namelist())

                with tqdm(total=total_files, desc="Extracting") as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)

            logger.info(f"✓ Extracted to {extract_to}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False

    def download_dataset(self):
        """
        Download SUN RGB-D dataset

        Note: This is a large dataset (~18GB). For testing, we'll download
        metadata and toolbox first, then provide instructions for full download.
        """
        logger.info("\n" + "="*60)
        logger.info("SUN RGB-D Dataset Download")
        logger.info("="*60 + "\n")

        logger.info("⚠️  WARNING: SUN RGB-D is a large dataset (~18GB compressed)")
        logger.info("   This script will guide you through the download process.\n")

        # Download toolbox (small)
        toolbox_zip = self.output_dir / 'SUNRGBDtoolbox.zip'
        if self.download_file(
            SUNRGBD_URLS['toolbox'],
            toolbox_zip,
            'SUN RGB-D Toolbox'
        ):
            toolbox_dir = self.output_dir / 'SUNRGBDtoolbox'
            if not toolbox_dir.exists():
                self.extract_zip(toolbox_zip, self.output_dir)

        # Download metadata
        metadata_file = self.output_dir / 'SUNRGBDMeta3DBB_v2.mat'
        self.download_file(
            SUNRGBD_URLS['metadata'],
            metadata_file,
            'SUN RGB-D Metadata'
        )

        # Main dataset - provide instructions
        logger.info("\n" + "="*60)
        logger.info("Main Dataset Download")
        logger.info("="*60)
        logger.info("\nTo download the full SUN RGB-D dataset:")
        logger.info(f"  1. Visit: {SUNRGBD_URLS['data_v1']}")
        logger.info(f"  2. Download SUNRGBD.zip (~18GB)")
        logger.info(f"  3. Place it in: {self.output_dir}")
        logger.info(f"  4. Run this script again with --extract flag\n")

        # Check if dataset already exists
        dataset_zip = self.output_dir / 'SUNRGBD.zip'
        dataset_dir = self.output_dir / 'SUNRGBD'

        if dataset_zip.exists() and not dataset_dir.exists():
            logger.info("✓ Found SUNRGBD.zip - extracting...")
            self.extract_zip(dataset_zip, self.output_dir)
        elif dataset_dir.exists():
            logger.info("✓ SUN RGB-D dataset already extracted")
        else:
            logger.warning("✗ SUNRGBD.zip not found. Please download manually.")
            return False

        return True

    def filter_indoor_scenes(self):
        """
        Filter dataset for indoor scenes

        Note: This requires scipy to read .mat files
        """
        try:
            import scipy.io
        except ImportError:
            logger.error("scipy is required to read .mat files")
            logger.info("Install with: pip install scipy")
            return []

        metadata_file = self.output_dir / 'SUNRGBDMeta3DBB_v2.mat'

        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return []

        logger.info("Loading SUN RGB-D metadata...")

        try:
            mat_data = scipy.io.loadmat(str(metadata_file))
            # The metadata structure varies - this is a simplified version
            # In reality, you'd need to parse the complex MATLAB structure

            logger.info("✓ Metadata loaded")

            # For now, return empty list with instructions
            logger.info("\n" + "="*60)
            logger.info("SUN RGB-D Filtering")
            logger.info("="*60)
            logger.info("\nFiltering requires parsing complex MATLAB metadata.")
            logger.info("Recommended approach:")
            logger.info("  1. Use MATLAB or Python scipy to load SUNRGBDMeta3DBB_v2.mat")
            logger.info("  2. Filter scenes by 'scene' field")
            logger.info(f"  3. Target scenes: {', '.join(sorted(TARGET_SCENE_TYPES))}")
            logger.info("  4. Copy filtered RGB and Depth images to output folders\n")

            return []

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []

    def process_dataset(self):
        """
        Process SUN RGB-D dataset

        This includes:
        - Filtering indoor scenes
        - Organizing RGB and depth images
        - Converting annotations
        """
        logger.info("\n" + "="*60)
        logger.info("Processing SUN RGB-D Dataset")
        logger.info("="*60 + "\n")

        dataset_dir = self.output_dir / 'SUNRGBD'

        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            logger.info("Please download and extract the dataset first.")
            return

        # Filter scenes
        filtered_scenes = self.filter_indoor_scenes()

        # Generate instructions for manual processing
        self.generate_processing_guide()

    def generate_processing_guide(self):
        """Generate guide for manually processing SUN RGB-D"""
        guide_file = self.output_dir / 'PROCESSING_GUIDE.md'

        guide_content = """# SUN RGB-D Processing Guide

## Overview
This guide explains how to process the SUN RGB-D dataset for BlindSpot training.

## Dataset Structure
```
SUNRGBD/
├── kv1/         # Kinect v1 data
├── kv2/         # Kinect v2 data
├── realsense/   # RealSense data
└── xtion/       # Xtion data
```

Each sequence contains:
- `image/`: RGB images
- `depth/`: Depth maps
- `annotation/`: 2D/3D bounding boxes

## Filtering Steps

### 1. Load Metadata
```python
import scipy.io
metadata = scipy.io.loadmat('SUNRGBDMeta3DBB_v2.mat')
```

### 2. Filter Indoor Scenes
Target scene types:
- bedroom
- bathroom
- living_room
- kitchen
- dining_room
- office
- home_office
- corridor
- hallway
- stairs

### 3. Copy Filtered Data
For each filtered scene:
```bash
# Copy RGB image
cp SUNRGBD/<sensor>/<scene_id>/image/*.jpg images/<scene_id>.jpg

# Copy depth map
cp SUNRGBD/<sensor>/<scene_id>/depth/*.png depth/<scene_id>.png
```

### 4. Convert Annotations
SUN RGB-D provides 2D bounding boxes in their annotation files.
Convert these to YOLO format:
```
class_id x_center y_center width height
```

## Estimated Output
- ~5,000-7,000 indoor scenes
- RGB images: 640x480 or similar
- Depth maps: same resolution
- Annotations: YOLO format

## Alternative: Use Pre-filtered Subset
For faster setup, you can:
1. Manually select 500-1000 representative indoor scenes
2. Focus on scenes with stairs, furniture, doors
3. Ensure variety in lighting conditions

## Next Steps
After processing:
```bash
python data_collection/verify_sun_rgbd.py
```

This will verify the processed data is ready for training.
"""

        with open(guide_file, 'w') as f:
            f.write(guide_content)

        logger.info(f"✓ Processing guide saved to {guide_file}")

    def generate_statistics(self):
        """Generate dataset statistics"""
        stats_file = self.output_dir / 'sun_rgbd_statistics.json'

        # Count processed images
        num_images = len(list(self.images_dir.glob('*.jpg'))) + \
                     len(list(self.images_dir.glob('*.png')))
        num_depth = len(list(self.depth_dir.glob('*.png')))
        num_annotations = len(list(self.annotations_dir.glob('*.txt')))

        self.stats.update({
            'processed_images': num_images,
            'depth_maps': num_depth,
            'annotations': num_annotations
        })

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("SUN RGB-D Statistics")
        logger.info("="*60)
        logger.info(f"Processed RGB images: {num_images}")
        logger.info(f"Depth maps: {num_depth}")
        logger.info(f"Annotations: {num_annotations}")
        logger.info(f"\n✓ Statistics saved to {stats_file}")

    def run(self):
        """Run complete download and processing pipeline"""
        logger.info("="*60)
        logger.info("SUN RGB-D Dataset Setup for BlindSpot")
        logger.info("="*60)

        # Step 1: Download dataset
        if not self.download_dataset():
            logger.warning("\nDataset download incomplete.")
            logger.info("Follow the instructions above to complete the download.")
            return

        # Step 2: Process dataset
        self.process_dataset()

        # Step 3: Generate statistics
        self.generate_statistics()

        logger.info("\n" + "="*60)
        logger.info("SUN RGB-D Setup Complete")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review PROCESSING_GUIDE.md")
        logger.info("2. Process the dataset manually or with custom scripts")
        logger.info("3. Verify with: python data_collection/verify_sun_rgbd.py")


def main():
    parser = argparse.ArgumentParser(
        description='Download and process SUN RGB-D dataset for BlindSpot'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/sun_rgbd',
        help='Output directory for SUN RGB-D data'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract SUNRGBD.zip if present'
    )

    args = parser.parse_args()

    # Create downloader and run
    downloader = SUNRGBDDownloader(args.output_dir)
    downloader.run()


if __name__ == '__main__':
    main()
