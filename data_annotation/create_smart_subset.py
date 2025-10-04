"""
Create intelligent subset for faster training without quality loss.
Selects most informative images based on:
- Object count (2-5 optimal)
- Class diversity
- Object sizes
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
import argparse

def score_image(label_path):
    """Score an image based on training value."""
    if not label_path.exists():
        return 0, []

    with open(label_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return 0, []

    num_objects = len(lines)
    unique_classes = len(set(line.split()[0] for line in lines))

    # Parse bounding box sizes
    box_sizes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            width = float(parts[3])
            height = float(parts[4])
            box_sizes.append(width * height)

    avg_box_size = sum(box_sizes) / len(box_sizes) if box_sizes else 0

    # Scoring system
    score = 0

    # Prefer 2-5 objects (not too sparse, not too crowded)
    if 2 <= num_objects <= 5:
        score += 15
    elif num_objects == 1:
        score += 5
    elif 6 <= num_objects <= 8:
        score += 10
    else:
        score += 3

    # Class diversity bonus (more classes = more learning)
    score += unique_classes * 6

    # Prefer medium-sized objects (not too small/large)
    if 0.05 <= avg_box_size <= 0.4:
        score += 10
    elif 0.02 <= avg_box_size <= 0.6:
        score += 5

    # Bonus for having objects
    if num_objects > 0:
        score += 5

    return score, lines

def create_smart_subset(source_dir, target_dir, train_count=3000, val_ratio=0.15, test_ratio=0.1, seed=42):
    """
    Create optimized subset for M2 Pro training.

    Args:
        source_dir: Original YOLO format directory
        target_dir: Output directory for subset
        train_count: Number of training images to select
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    print(f"\n{'='*70}")
    print(f"Creating Smart Subset for Optimized M2 Pro Training")
    print(f"{'='*70}\n")

    # Process each split
    for split in ['train', 'val', 'test']:
        images_dir = source_dir / 'images' / split
        labels_dir = source_dir / 'labels' / split

        if not images_dir.exists():
            print(f"⚠️  {split} directory not found, skipping...")
            continue

        print(f"📊 Processing {split} split...")

        # Score all images
        image_scores = []
        for img_path in images_dir.glob('*.jpg'):
            label_path = labels_dir / img_path.with_suffix('.txt').name
            score, lines = score_image(label_path)

            if score > 0:  # Only include images with objects
                image_scores.append((img_path, label_path, score, len(lines)))

        print(f"   Found {len(image_scores)} valid images")

        # Sort by score (highest quality first)
        image_scores.sort(key=lambda x: x[2], reverse=True)

        # Determine subset size
        if split == 'train':
            subset_size = train_count
        elif split == 'val':
            subset_size = int(train_count * val_ratio)
        else:  # test
            subset_size = int(train_count * test_ratio)

        # Select top images
        selected = image_scores[:min(subset_size, len(image_scores))]

        print(f"   Selected {len(selected)} images (avg score: {sum(s for _, _, s, _ in selected) / len(selected):.1f})")

        # Create target directories
        target_imgs = target_dir / 'images' / split
        target_labels = target_dir / 'labels' / split
        target_imgs.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        # Copy selected files
        for img_path, label_path, score, obj_count in selected:
            shutil.copy(img_path, target_imgs / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, target_labels / label_path.name)

        print(f"   ✅ Copied to {target_imgs.parent}")

    # Copy dataset.yaml
    yaml_src = source_dir / 'dataset.yaml'
    yaml_dst = target_dir / 'dataset.yaml'
    if yaml_src.exists():
        # Update paths in yaml
        with open(yaml_src) as f:
            content = f.read()

        # Replace path
        content = content.replace(str(source_dir), str(target_dir))

        with open(yaml_dst, 'w') as f:
            f.write(content)

        print(f"\n✅ Dataset config copied to {yaml_dst}")

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"Subset Creation Complete!")
    print(f"{'='*70}")
    print(f"📁 Location: {target_dir}")
    print(f"📊 Train: {train_count} images")
    print(f"📊 Val: {int(train_count * val_ratio)} images")
    print(f"📊 Test: {int(train_count * test_ratio)} images")
    print(f"🎯 Quality: High-scoring images selected")
    print(f"⚡ Optimized for: M2 Pro local training")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create smart dataset subset')
    parser.add_argument('--source', type=str, default='data/processed/yolo_format',
                        help='Source YOLO format directory')
    parser.add_argument('--target', type=str, default='data/processed/yolo_subset_3k',
                        help='Target directory for subset')
    parser.add_argument('--train-count', type=int, default=3000,
                        help='Number of training images')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                        help='Test set ratio')

    args = parser.parse_args()

    create_smart_subset(
        args.source,
        args.target,
        args.train_count,
        args.val_ratio,
        args.test_ratio
    )
