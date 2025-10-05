#!/usr/bin/env python3
"""
Download the LARGER version 11 of indoor obstacle detection dataset.
"""

from roboflow import Roboflow
import os
from pathlib import Path

api_key = 'Uz93xVUYB2g7LhkbShT0'

print("="*70)
print("Downloading LARGER Indoor Obstacle Dataset (v11)")
print("="*70)

output_dir = Path("data/raw/roboflow")
output_dir.mkdir(parents=True, exist_ok=True)

rf = Roboflow(api_key=api_key)

# Download Dataset 1 VERSION 11 (1440 images instead of 293!)
print("\n[1/2] Downloading: Indoor Obstacle Detection v11...")
print("  - 1440 train images (vs 270 in v1)")
print("  - 57 validation images")
print("  - 54 test images")

try:
    project1 = rf.workspace("indooroutdoornavigation").project("indoor-obstacle-detection")
    dataset1 = project1.version(11).download("yolov8", location=str(output_dir / "dataset1_v11"))
    print("✓ Dataset 1 v11 downloaded!")
except Exception as e:
    print(f"⚠️  Error downloading v11: {e}")
    print("  Using existing v1 dataset")

# Dataset 2 is already good (156 images)
print("\n[2/2] Dataset 2 already downloaded (156 images)")

print("\n" + "="*70)
print("✓ Download complete!")
print("="*70)
