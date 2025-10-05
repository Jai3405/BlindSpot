#!/usr/bin/env python3
"""
Fine-tune existing COCO-trained model on indoor obstacle datasets.
Uses transfer learning for faster training and better results.
"""

import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

print("="*70)
print("Fine-Tuning COCO Model on Indoor Obstacles")
print("Transfer Learning: COCO → Indoor Navigation")
print("="*70)

# Check for downloaded datasets
roboflow_dir = Path("data/raw/roboflow")
dataset1 = roboflow_dir / "dataset1"
dataset2 = roboflow_dir / "dataset2"

if not dataset1.exists() and not dataset2.exists():
    print("\n⚠️  No Roboflow datasets found!")
    print("\nPlease download datasets first:")
    print("1. Visit: https://universe.roboflow.com/indooroutdoornavigation/indoor-obstacle-detection")
    print("2. Click 'Download' → Format: YOLOv8")
    print("3. Extract to: data/raw/roboflow/dataset1/")
    print("")
    print("4. Visit: https://universe.roboflow.com/project-wcqez/indoor-obstacles")
    print("5. Click 'Download' → Format: YOLOv8")
    print("6. Extract to: data/raw/roboflow/dataset2/")
    exit(1)

# Use existing COCO-trained model as starting point
base_model = "runs/train/blindspot_optimized/weights/best.pt"

if not Path(base_model).exists():
    print(f"\n⚠️  Base model not found: {base_model}")
    print("Using pretrained YOLOv8n instead")
    base_model = "models/pretrained/yolov8n.pt"

print(f"\n📦 Base model: {base_model}")

# Find data.yaml from downloaded datasets
data_yaml = None
if (dataset1 / "data.yaml").exists():
    data_yaml = dataset1 / "data.yaml"
elif (dataset2 / "data.yaml").exists():
    data_yaml = dataset2 / "data.yaml"

if data_yaml:
    print(f"📊 Dataset config: {data_yaml}")
    
    # Load model
    model = YOLO(base_model)
    
    # Training configuration (optimized for M2 Pro)
    print("\n🚀 Starting fine-tuning...")
    print("   - Using existing weights as starting point")
    print("   - Training on indoor obstacles")
    print("   - Optimized for M2 Pro")
    
    results = model.train(
        data=str(data_yaml),
        epochs=30,              # Fewer epochs for fine-tuning
        imgsz=416,             # Smaller for speed
        batch=16,
        cache=True,            # RAM caching
        device='cpu',          # CPU for stability
        patience=10,           # Early stopping
        project='runs/finetune',
        name='indoor_obstacles',
        exist_ok=True,
        
        # Fine-tuning specific
        freeze=10,             # Freeze first 10 layers (keep COCO knowledge)
        lr0=0.001,            # Lower learning rate for fine-tuning
        
        # Augmentation (minimal for fine-tuning)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    print("\n" + "="*70)
    print("✓ Fine-tuning complete!")
    print(f"  Model saved: runs/finetune/indoor_obstacles/weights/best.pt")
    print("\nNext: Test with demo_blindspot.py using new model")
    print("="*70)
else:
    print("\n⚠️  Could not find data.yaml in downloaded datasets")
    print("Please ensure datasets are properly extracted")

