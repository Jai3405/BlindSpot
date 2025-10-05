# BlindSpot AI - Project Structure

## Overview
Clean, organized structure for the BlindSpot AI assistive navigation system.

## Directory Structure

```
BlindSpot/
├── config/                    # Configuration files
│   └── data_config.yaml      # Dataset configuration
│
├── data/                      # Dataset storage (gitignored)
│   ├── raw/coco/             # COCO dataset (8.8GB)
│   └── processed/            # Processed datasets
│       └── yolo_subset_3k/   # Smart subset for training (1.2GB)
│
├── data_collection/           # Data collection scripts
│   ├── download_coco.py      # COCO dataset downloader
│   └── verify_dataset.py     # Dataset verification
│
├── data_annotation/           # Data processing tools
│   ├── convert_to_yolo.py    # COCO to YOLO format
│   └── create_smart_subset.py # Intelligent subset creation
│
├── model_training/            # Training scripts
│   └── train_yolov8.py       # YOLOv8 training pipeline
│
├── inference/                 # Inference system
│   ├── blindspot_engine.py   # Main inference engine
│   ├── depth_estimator.py    # MiDaS depth estimation
│   ├── spatial_analyzer.py   # Spatial analysis
│   └── audio_feedback.py     # TTS audio system
│
├── runs/                      # Training outputs (gitignored)
│   └── train/
│       └── blindspot_optimized/  # Best trained model (93MB)
│
├── models/                    # Model storage
│   └── pretrained/           # Pretrained models
│       └── yolov8n.pt        # YOLOv8-nano base (6.2MB)
│
├── logs/                      # Training logs (gitignored)
│   ├── coco_download.log
│   ├── training_optimized.log
│   └── ...
│
├── docs/                      # Documentation
│   └── DATA_COLLECTION.md
│
├── scripts/                   # Utility scripts
│   └── setup_environment.sh
│
├── demo_blindspot.py          # Complete demo application
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

## Key Files

### Training Pipeline
- `data_collection/download_coco.py` - Downloads COCO dataset
- `data_annotation/create_smart_subset.py` - Creates optimized 3k subset
- `model_training/train_yolov8.py` - Trains YOLOv8 model

### Inference System
- `inference/blindspot_engine.py` - Main inference pipeline
- `demo_blindspot.py` - Demo application (webcam/video/image)

### Trained Model
- `runs/train/blindspot_optimized/weights/best.pt` - YOLOv8n (46.7% mAP@0.5)

## Data Sizes

| Component | Size |
|-----------|------|
| COCO raw data | 8.8 GB |
| Training subset | 1.2 GB |
| Trained model | 93 MB |
| Pretrained YOLOv8n | 6.2 MB |
| **Total** | ~10 GB |

## Usage

```bash
# Run demo with webcam
python demo_blindspot.py --mode webcam

# Process video
python demo_blindspot.py --mode video --input video.mp4

# Process image
python demo_blindspot.py --mode image --input image.jpg
```

## Clean Workspace

All temporary files, logs, and large datasets are gitignored. Only essential code and trained models are tracked in git.
