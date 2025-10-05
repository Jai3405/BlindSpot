#!/usr/bin/env python3
"""
Fine-tune COCO model on COMBINED indoor obstacle datasets.
Uses BOTH Roboflow datasets for maximum coverage.
"""

from pathlib import Path
from ultralytics import YOLO

print("="*70)
print("Fine-Tuning on COMBINED Indoor Datasets")
print("Transfer Learning: COCO → Indoor (Both Datasets)")
print("="*70)

# Use combined dataset
data_yaml = Path("data/processed/indoor_combined/data.yaml")

if not data_yaml.exists():
    print(f"\n⚠️  Combined dataset not found: {data_yaml}")
    print("Run: venv/bin/python data_annotation/merge_roboflow_datasets.py")
    exit(1)

print(f"\n📊 Dataset: {data_yaml}")
print(f"   - 469 images (426 train + 22 val + 21 test)")
print(f"   - 11 classes: door, wall, stairs, person, obstacle, etc.")

# Load COCO-trained model
base_model = "runs/train/blindspot_optimized/weights/best.pt"
print(f"\n📦 Base model: {base_model}")

model = YOLO(base_model)

# Fine-tuning configuration
print("\n🚀 Starting fine-tuning...")
print("   ✓ Transfer learning from COCO model")
print("   ✓ Training on BOTH indoor datasets combined")
print("   ✓ Optimized for M2 Pro CPU")

results = model.train(
    data=str(data_yaml),
    epochs=30,
    imgsz=416,
    batch=16,
    cache=True,
    device='cpu',
    patience=10,
    project='runs/finetune',
    name='indoor_combined',
    exist_ok=True,
    
    # Transfer learning settings
    freeze=10,             # Keep COCO knowledge
    lr0=0.001,            # Lower LR for fine-tuning
    
    # Minimal augmentation
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
)

print("\n" + "="*70)
print("✓ Fine-tuning complete!")
print(f"  Model: runs/finetune/indoor_combined/weights/best.pt")
print("\nNew capabilities:")
print("  ✅ Doors (all types)")
print("  ✅ Walls")
print("  ✅ Stairs (CRITICAL!)")
print("  ✅ Obstacles")
print("  ✅ Person")
print("  ✅ Furniture (chair, desk)")
print("  ✅ Elevator")
print("\nTest with:")
print("  ./venv/bin/python demo_blindspot.py \\")
print("    --mode webcam \\")
print("    --model runs/finetune/indoor_combined/weights/best.pt")
print("="*70)
