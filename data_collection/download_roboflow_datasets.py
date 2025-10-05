#!/usr/bin/env python3
"""
Download both Roboflow indoor obstacle datasets and prepare for fine-tuning.
"""

from roboflow import Roboflow
import os
from pathlib import Path

print("="*70)
print("Downloading Roboflow Indoor Obstacle Datasets")
print("="*70)

# You'll need to get your API key from: https://app.roboflow.com/settings/api
print("\n📋 To download, you need a Roboflow API key (free account)")
print("   1. Go to: https://app.roboflow.com/settings/api")
print("   2. Copy your API key")
print("   3. Set it as environment variable or paste when prompted")
print()

api_key = os.getenv('ROBOFLOW_API_KEY')

if not api_key:
    print("⚠️  ROBOFLOW_API_KEY not found in environment")
    print("\nOption 1: Set environment variable:")
    print("   export ROBOFLOW_API_KEY='your_api_key_here'")
    print("\nOption 2: Manual download:")
    print("   1. Visit the dataset URLs")
    print("   2. Click 'Download Dataset'")
    print("   3. Select 'YOLOv8' format")
    print("   4. Download to data/raw/roboflow/")
    print()
    print("Dataset 1: https://universe.roboflow.com/indooroutdoornavigation/indoor-obstacle-detection")
    print("Dataset 2: https://universe.roboflow.com/project-wcqez/indoor-obstacles")
    exit(0)

# Create directory
output_dir = Path("data/raw/roboflow")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📥 Downloading to: {output_dir}")

# Download Dataset 1: Indoor Obstacle Detection
print("\n[1/2] Downloading: Indoor Obstacle Detection...")
rf = Roboflow(api_key=api_key)
project1 = rf.workspace("indooroutdoornavigation").project("indoor-obstacle-detection")
dataset1 = project1.version(1).download("yolov8", location=str(output_dir / "dataset1"))
print("✓ Dataset 1 downloaded!")

# Download Dataset 2: Indoor Obstacles  
print("\n[2/2] Downloading: Indoor Obstacles...")
project2 = rf.workspace("project-wcqez").project("indoor-obstacles")
dataset2 = project2.version(1).download("yolov8", location=str(output_dir / "dataset2"))
print("✓ Dataset 2 downloaded!")

print("\n" + "="*70)
print("✓ Both datasets downloaded successfully!")
print(f"  Location: {output_dir}")
print("\nNext step: Merge datasets and fine-tune COCO model")
print("="*70)
