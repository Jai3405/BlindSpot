#!/usr/bin/env python3
"""
Merge both Roboflow datasets for comprehensive indoor obstacle detection.
"""

import yaml
import shutil
from pathlib import Path

print("="*70)
print("Merging Roboflow Datasets")
print("="*70)

# Create merged dataset directory
merged_dir = Path("data/processed/indoor_combined")
merged_dir.mkdir(parents=True, exist_ok=True)

# Read both data.yaml files
dataset1_yaml = Path("data/raw/roboflow/dataset1/data.yaml")
dataset2_yaml = Path("data/raw/roboflow/dataset2/data.yaml")

with open(dataset1_yaml) as f:
    data1 = yaml.safe_load(f)
    
with open(dataset2_yaml) as f:
    data2 = yaml.safe_load(f)

print(f"\nDataset 1 classes: {data1['names']}")
print(f"Dataset 2 classes: {data2['names']}")

# Map similar classes together
class_mapping = {
    'Door': 'door',  # Generic door
    'Wall': 'wall',
    'person': 'person',
    'closed_door': 'door',
    'opened_door': 'door',
    'obstacle': 'obstacle',
    'Low Obstacle': 'obstacle',
    'unknown obstacle': 'obstacle',
    'Chair': 'chair',
    'Desk': 'desk',
    'stairs': 'stairs',
    'Elevator': 'elevator',
    'footpath': 'footpath',
    'sign': 'sign',
    'switch': 'switch'
}

# Build unified class list
unified_classes = list(set(class_mapping.values()))
unified_classes.sort()

print(f"\nUnified classes ({len(unified_classes)}):")
for i, cls in enumerate(unified_classes):
    print(f"  {i}: {cls}")

# Create class ID mapping for both datasets
ds1_class_map = {}
for i, cls in enumerate(data1['names']):
    unified_cls = class_mapping.get(cls, cls)
    ds1_class_map[i] = unified_classes.index(unified_cls)

ds2_class_map = {}
for i, cls in enumerate(data2['names']):
    unified_cls = class_mapping.get(cls, cls)
    ds2_class_map[i] = unified_classes.index(unified_cls)

# Create merged data.yaml
merged_yaml = {
    'path': str(merged_dir.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(unified_classes),
    'names': unified_classes
}

with open(merged_dir / 'data.yaml', 'w') as f:
    yaml.dump(merged_yaml, f)

print(f"\n✓ Created merged data.yaml")
print(f"  Location: {merged_dir / 'data.yaml'}")

# Copy and merge images/labels
total_images = 0
for split in ['train', 'val', 'test']:
    print(f"\nMerging {split} split...")
    
    # Create directories
    (merged_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (merged_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    split_count = 0
    
    # Dataset 1
    d1_imgs = Path(f"data/raw/roboflow/dataset1/{split if split != 'val' else 'valid'}/images")
    d1_labels = Path(f"data/raw/roboflow/dataset1/{split if split != 'val' else 'valid'}/labels")
    
    if d1_imgs.exists():
        for img in d1_imgs.glob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, merged_dir / split / 'images' / f"ds1_{img.name}")
                
                label = d1_labels / f"{img.stem}.txt"
                if label.exists():
                    with open(label) as f:
                        lines = f.readlines()
                    
                    # Remap class IDs
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            old_cls_id = int(parts[0])
                            new_cls_id = ds1_class_map[old_cls_id]
                            new_lines.append(f"{new_cls_id} {' '.join(parts[1:])}\n")
                    
                    with open(merged_dir / split / 'labels' / f"ds1_{img.stem}.txt", 'w') as f:
                        f.writelines(new_lines)
                split_count += 1
    
    # Dataset 2
    d2_imgs = Path(f"data/raw/roboflow/dataset2/{split if split != 'val' else 'valid'}/images")
    d2_labels = Path(f"data/raw/roboflow/dataset2/{split if split != 'val' else 'valid'}/labels")
    
    if d2_imgs.exists():
        for img in d2_imgs.glob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, merged_dir / split / 'images' / f"ds2_{img.name}")
                
                label = d2_labels / f"{img.stem}.txt"
                if label.exists():
                    with open(label) as f:
                        lines = f.readlines()
                    
                    # Remap class IDs
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            old_cls_id = int(parts[0])
                            new_cls_id = ds2_class_map[old_cls_id]
                            new_lines.append(f"{new_cls_id} {' '.join(parts[1:])}\n")
                    
                    with open(merged_dir / split / 'labels' / f"ds2_{img.stem}.txt", 'w') as f:
                        f.writelines(new_lines)
                split_count += 1
    
    print(f"  ✓ {split}: {split_count} images")
    total_images += split_count

print("\n" + "="*70)
print("✓ Datasets merged successfully!")
print(f"  Combined dataset: {merged_dir}")
print(f"  Total classes: {len(unified_classes)}")
print(f"  Total images: {total_images}")
print("\nClasses:")
for cls in unified_classes:
    print(f"  - {cls}")
print("="*70)
