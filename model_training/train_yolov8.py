"""
YOLOv8 Training Script for BlindSpot
Train object detection model for indoor obstacle detection
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch


def train_blindspot_model(
    data_yaml: str = 'data/processed/yolo_format/dataset.yaml',
    model_size: str = 'n',  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    resume: bool = False,
    project: str = 'runs/train',
    name: str = 'blindspot_yolov8',
    cache: bool = False,
    workers: int = 8
):
    """
    Train YOLOv8 model on BlindSpot dataset

    Args:
        data_yaml: Path to dataset.yaml
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to use (None=auto, 'cpu', '0', '0,1', etc.)
        resume: Resume training from last checkpoint
        project: Project directory
        name: Run name
    """

    print("="*70)
    print("BlindSpot YOLOv8 Training")
    print("="*70)

    # Auto-detect device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}\n")

    # Load pre-trained YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        resume=resume,

        # Performance settings
        patience=15,  # Early stopping patience (optimized)
        save=True,    # Save checkpoints
        save_period=10,  # Save every N epochs
        cache=cache,  # Cache images for faster training

        # Augmentation (optimized for speed without quality loss)
        augment=True,
        hsv_h=0.01,   # Light hue augmentation
        hsv_s=0.5,    # Light saturation augmentation
        hsv_v=0.3,    # Light value augmentation
        degrees=0.0,  # No rotation (slow on CPU)
        translate=0.05,  # Minimal translation
        scale=0.3,    # Reduced scale augmentation
        shear=0.0,    # No shear (slow on CPU)
        perspective=0.0,  # No perspective (slow on CPU)
        flipud=0.0,   # No vertical flip
        fliplr=0.5,   # Keep horizontal flip
        mosaic=0.3,   # Reduced mosaic (expensive augmentation)
        mixup=0.0,    # No mixup (expensive)

        # Training hyperparameters
        lr0=0.01,     # Initial learning rate
        lrf=0.01,     # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Other settings
        workers=workers,  # Number of data loader workers
        verbose=True,
        seed=42,      # Random seed for reproducibility
    )

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nResults saved to: {project}/{name}")
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"Last weights: {project}/{name}/weights/last.pt")

    return results


def validate_model(weights: str, data_yaml: str, device: str = None):
    """
    Validate trained model

    Args:
        weights: Path to trained weights
        data_yaml: Path to dataset.yaml
        device: Device to use
    """
    print("\n" + "="*70)
    print("Model Validation")
    print("="*70)

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = YOLO(weights)

    results = model.val(
        data=data_yaml,
        device=device,
        batch=16,
        imgsz=640,
        split='val',
        verbose=True
    )

    print("\n" + "="*70)
    print("Validation Results")
    print("="*70)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    return results


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 on BlindSpot dataset')
    parser.add_argument('--data', type=str, default='data/processed/yolo_format/dataset.yaml',
                       help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (None=auto, cpu, 0, 0,1, etc.)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--validate-only', type=str, default=None,
                       help='Only validate using provided weights')
    parser.add_argument('--name', type=str, default='blindspot_yolov8',
                       help='Run name')
    parser.add_argument('--cache', action='store_true',
                       help='Cache images in RAM for faster training')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers')

    args = parser.parse_args()

    if args.validate_only:
        # Validation only
        validate_model(args.validate_only, args.data, args.device)
    else:
        # Train model
        results = train_blindspot_model(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            resume=args.resume,
            name=args.name,
            cache=args.cache,
            workers=args.workers
        )

        # Validate best model
        best_weights = f"runs/train/{args.name}/weights/best.pt"
        if Path(best_weights).exists():
            print("\n" + "="*70)
            print("Validating Best Model")
            print("="*70)
            validate_model(best_weights, args.data, args.device)


if __name__ == '__main__':
    main()
