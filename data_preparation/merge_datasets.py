#!/usr/bin/env python3
"""
Dataset Merger for BlindSpot AI
Merges COCO subset (3000 images) + Indoor Obstacles v11 (1440 images)
Properly maintains COCO knowledge while adding indoor navigation capabilities.
"""

import yaml
import shutil
from pathlib import Path
from collections import defaultdict
import random

print("="*70)
print("BlindSpot Dataset Merger")
print("="*70)

# Source datasets
COCO_PATH = Path("/Volumes/Jaisung/BlindSpot/data/processed/yolo_subset_3k")
INDOOR_PATH = Path("/Volumes/Jaisung/BlindSpot/data/raw/roboflow/dataset1_v11")

# Output path
OUTPUT_PATH = Path("/Volumes/Jaisung/BlindSpot/data/processed/merged_navigation")

# Load source configs
with open(COCO_PATH / "dataset.yaml") as f:
    coco_config = yaml.safe_load(f)

with open(INDOOR_PATH / "data.yaml") as f:
    indoor_config = yaml.safe_load(f)

print(f"\n📊 Source Datasets:")
print(f"  COCO Subset: {len(coco_config['names'])} classes")
print(f"  Indoor v11: {indoor_config['nc']} classes")

# Define merged class mapping
# Strategy: COCO classes first (0-16), then NEW indoor classes (17-23)

MERGED_CLASSES = [
    # COCO classes (0-16) - keep as-is
    'broccoli',      # 0
    'carrot',        # 1
    'hot dog',       # 2
    'pizza',         # 3
    'donut',         # 4
    'cake',          # 5
    'chair',         # 6
    'couch',         # 7
    'potted plant',  # 8
    'bed',           # 9
    'dining table',  # 10
    'tv',            # 11
    'laptop',        # 12
    'mouse',         # 13
    'remote',        # 14
    'keyboard',      # 15
    'cell phone',    # 16

    # NEW indoor navigation classes (17-23)
    'door',          # 17 (merged: door + closed_door from indoor)
    'wall',          # 18
    'obstacle',      # 19
    'elevator',      # 20
    'escalator',     # 21
    'footpath',      # 22
    'person',        # 23
]

# Class ID mappings
# COCO: 0-16 stay the same
coco_class_map = {i: i for i in range(17)}

# Indoor: remap to 17-23
indoor_class_map = {
    indoor_config['names'].index('door'): 17,
    indoor_config['names'].index('closed_door'): 17,  # Merge closed_door → door
    indoor_config['names'].index('wall'): 18,
    indoor_config['names'].index('obstacle'): 19,
    indoor_config['names'].index('elevator'): 20,
    indoor_config['names'].index('escalator'): 21,
    indoor_config['names'].index('footpath'): 22,
    indoor_config['names'].index('person'): 23,
}

print(f"\n🔗 Merged Classes: {len(MERGED_CLASSES)} total")
print(f"  COCO classes: 0-16 (17 classes)")
print(f"  Indoor classes: 17-23 (7 new classes)")
print(f"  Total: {len(MERGED_CLASSES)} classes\n")

# Create output structure
for split in ['train', 'val', 'test']:
    (OUTPUT_PATH / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / split / 'labels').mkdir(parents=True, exist_ok=True)

print("📁 Created output directory structure\n")

# Helper function to convert label file
def convert_label(label_path, class_map):
    """Convert label file with class ID remapping."""
    if not label_path.exists():
        return []

    converted_lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            old_class_id = int(parts[0])
            if old_class_id not in class_map:
                continue  # Skip unmapped classes

            new_class_id = class_map[old_class_id]
            parts[0] = str(new_class_id)
            converted_lines.append(' '.join(parts))

    return converted_lines

# Statistics
stats = defaultdict(int)

print("🔄 Processing COCO dataset...")

# Process COCO dataset (keep all splits as-is)
for split in ['train', 'val', 'test']:
    coco_img_dir = COCO_PATH / 'images' / split
    coco_lbl_dir = COCO_PATH / 'labels' / split

    if not coco_img_dir.exists():
        continue

    images = list(coco_img_dir.glob('*.[jJ][pP][gG]')) + \
             list(coco_img_dir.glob('*.[pP][nN][gG]'))

    for img_path in images:
        # Copy image
        shutil.copy(img_path, OUTPUT_PATH / split / 'images' / f"coco_{img_path.name}")

        # Convert and copy label
        lbl_path = coco_lbl_dir / f"{img_path.stem}.txt"
        converted_labels = convert_label(lbl_path, coco_class_map)

        if converted_labels:
            with open(OUTPUT_PATH / split / 'labels' / f"coco_{img_path.stem}.txt", 'w') as f:
                f.write('\n'.join(converted_labels))

            # Count classes
            for line in converted_labels:
                class_id = int(line.split()[0])
                stats[f"coco_{split}_{MERGED_CLASSES[class_id]}"] += 1

        stats[f"coco_{split}_images"] += 1

    print(f"  ✓ COCO {split}: {stats[f'coco_{split}_images']} images")

print(f"\n🔄 Processing Indoor Obstacles v11 dataset...")

# Process Indoor dataset
# Map: train→train, valid→val, test→test
indoor_split_map = {
    'train': 'train',
    'valid': 'val',
    'test': 'test'
}

for indoor_split, merged_split in indoor_split_map.items():
    indoor_img_dir = INDOOR_PATH / indoor_split / 'images'
    indoor_lbl_dir = INDOOR_PATH / indoor_split / 'labels'

    if not indoor_img_dir.exists():
        continue

    images = list(indoor_img_dir.glob('*.[jJ][pP][gG]')) + \
             list(indoor_img_dir.glob('*.[pP][nN][gG]'))

    for img_path in images:
        # Copy image
        shutil.copy(img_path, OUTPUT_PATH / merged_split / 'images' / f"indoor_{img_path.name}")

        # Convert and copy label
        lbl_path = indoor_lbl_dir / f"{img_path.stem}.txt"
        converted_labels = convert_label(lbl_path, indoor_class_map)

        if converted_labels:
            with open(OUTPUT_PATH / merged_split / 'labels' / f"indoor_{img_path.stem}.txt", 'w') as f:
                f.write('\n'.join(converted_labels))

            # Count classes
            for line in converted_labels:
                class_id = int(line.split()[0])
                stats[f"indoor_{merged_split}_{MERGED_CLASSES[class_id]}"] += 1

        stats[f"indoor_{merged_split}_images"] += 1

    print(f"  ✓ Indoor {indoor_split}→{merged_split}: {stats[f'indoor_{merged_split}_images']} images")

# Create dataset.yaml
dataset_config = {
    'path': str(OUTPUT_PATH.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(MERGED_CLASSES),
    'names': {i: name for i, name in enumerate(MERGED_CLASSES)}
}

with open(OUTPUT_PATH / 'dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

print(f"\n✅ Dataset merge complete!")
print(f"\n📊 Final Statistics:")
print(f"  Output: {OUTPUT_PATH}")
print(f"\n  Split Breakdown:")

for split in ['train', 'val', 'test']:
    coco_count = stats.get(f"coco_{split}_images", 0)
    indoor_count = stats.get(f"indoor_{split}_images", 0)
    total = coco_count + indoor_count
    print(f"    {split.upper():5} - COCO: {coco_count:4} | Indoor: {indoor_count:4} | Total: {total:4}")

total_images = sum(stats[f"coco_{s}_images"] for s in ['train', 'val', 'test']) + \
               sum(stats[f"indoor_{s}_images"] for s in ['train', 'val', 'test'])

print(f"\n  Total Images: {total_images}")
print(f"  Total Classes: {len(MERGED_CLASSES)}")
print(f"\n  Class Distribution (combined train+val+test):")

# Aggregate class counts
class_totals = defaultdict(int)
for key, count in stats.items():
    if '_images' in key:
        continue
    for class_name in MERGED_CLASSES:
        if key.endswith(f"_{class_name}"):
            class_totals[class_name] += count

for i, class_name in enumerate(MERGED_CLASSES):
    count = class_totals.get(class_name, 0)
    source = "COCO" if i < 17 else "Indoor"
    print(f"    {i:2}: {class_name:15} - {count:4} instances ({source})")

print(f"\n✅ Created: {OUTPUT_PATH / 'dataset.yaml'}")
print("="*70)
