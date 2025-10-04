# BlindSpot 🦯

**AI-Powered Obstacle Detection System for Blind Navigation**

BlindSpot is an end-to-end computer vision system designed to help blind and visually impaired individuals navigate indoor environments safely. It combines YOLOv8 object detection with MiDaS depth estimation to provide real-time spatial awareness through audio feedback.

---

## 🎯 Project Overview

### Features

- **Real-time Obstacle Detection**: Identifies common indoor obstacles (furniture, people, stairs, doors)
- **Depth Estimation**: Calculates distance to obstacles using MiDaS
- **Spatial Audio Feedback**: Stereo-panned text-to-speech alerts based on obstacle position
- **Critical Hazard Detection**: Special focus on stairs, door states, and low obstacles
- **Complete Training Pipeline**: Full data collection, annotation, and training workflow from scratch

### Current Status: Phase 1 - Data Collection ✅

This repository currently includes:
- ✅ COCO dataset downloader with intelligent filtering
- ✅ SUN RGB-D dataset integration tools
- ✅ Video frame extraction with quality filtering
- ✅ Intelligent frame selection using clustering
- ✅ Comprehensive documentation

**Coming Next:** Phase 2 - Data Annotation Tools

---

## 📁 Project Structure

```
blindspot/
├── data_collection/          # Phase 1: Data gathering tools
│   ├── download_coco.py      # COCO dataset downloader
│   ├── download_sun_rgbd.py  # SUN RGB-D downloader
│   ├── video_to_frames.py    # Extract frames from videos
│   └── frame_selector.py     # Select diverse frames for annotation
│
├── data_annotation/          # Phase 2: Annotation tools (coming soon)
│   ├── coco_to_yolo.py       # Convert COCO → YOLO format
│   ├── verify_annotations.py # Validate annotations
│   └── augment_annotations.py # Generate augmented data
│
├── training/                 # Phase 3: Model training (coming soon)
│   ├── train_yolo.py         # YOLOv8 training script
│   ├── evaluate.py           # Model evaluation
│   └── export_model.py       # Export to ONNX/TensorRT
│
├── src/                      # Phase 4: Inference system (coming soon)
│   ├── detection.py          # YOLOv8 obstacle detector
│   ├── depth_estimation.py   # MiDaS depth maps
│   ├── spatial_analyzer.py   # 3D position calculator
│   ├── audio_feedback.py     # Text-to-speech + spatial audio
│   └── main.py               # Main application
│
├── data/                     # Dataset storage
│   ├── raw/                  # Raw collected data
│   │   ├── coco/             # COCO dataset
│   │   ├── sun_rgbd/         # SUN RGB-D dataset
│   │   └── custom/           # Custom recorded data
│   └── processed/            # Processed training data
│
├── config/                   # Configuration files
├── docs/                     # Documentation
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks for analysis
└── scripts/                  # Utility scripts
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM
- 50GB+ free disk space (for full dataset)
- (Optional) CUDA-capable GPU for training

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/blindspot.git
cd blindspot

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate environment
source venv/bin/activate
```

### Phase 1: Data Collection

#### 1. Download COCO Dataset

```bash
# Download annotations only (fast, ~500MB)
python data_collection/download_coco.py --no-images

# Or download with images (slow, ~18GB)
python data_collection/download_coco.py
```

**Expected output:**
- 15,000-20,000 filtered images with indoor obstacles
- Annotations in COCO JSON format
- Statistics file with class distribution

#### 2. Download SUN RGB-D Dataset (Optional)

```bash
# Download toolbox and metadata
python data_collection/download_sun_rgbd.py

# Follow instructions to download full dataset (~18GB)
# See: docs/DATA_COLLECTION.md
```

#### 3. Collect Custom Data

**Step 3a: Record Videos**

Record videos of indoor environments focusing on:
- Stairs (ascending/descending) - **Critical**
- Doorways (open/closed)
- Furniture and obstacles
- Various lighting conditions

See [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) for detailed recording guidelines.

**Step 3b: Extract Frames**

```bash
# Single video
python data_collection/video_to_frames.py \
    --video my_video.mp4 \
    --output data/raw/custom/frames \
    --fps 2.0

# Batch process
python data_collection/video_to_frames.py \
    --video-dir videos/ \
    --output data/raw/custom/frames
```

**Step 3c: Select Diverse Frames**

```bash
python data_collection/frame_selector.py \
    --input data/raw/custom/frames \
    --output data/raw/custom/selected \
    --num-frames 300
```

This intelligently selects 300 diverse frames for annotation using:
- Perceptual hashing for duplicate removal
- Feature extraction (color, texture, edges)
- K-means clustering
- Representative frame selection

---

## 📊 Dataset Overview

### Target Dataset Composition

| Source | Images | Purpose |
|--------|--------|---------|
| **COCO 2017** | 15,000-20,000 | Common indoor objects (furniture, people, appliances) |
| **SUN RGB-D** | 5,000-7,000 | Indoor scenes with depth information |
| **Custom** | 2,000-3,000 | Critical classes (stairs, doors, edges, low obstacles) |
| **Total** | **22,000-30,000** | Complete training dataset |

### Target Classes

#### From COCO (15 classes)
- person, chair, couch, bed, dining_table
- toilet, tv, laptop, potted_plant, refrigerator
- book, clock, vase, microwave, keyboard

#### Custom Classes (9 classes)
- stairs_up, stairs_down
- door_open, door_closed
- table_edge, low_obstacle, furniture_leg
- narrow_passage, glass_door

**Total: 24 classes**

---

## 📖 Documentation

- **[Data Collection Guide](docs/DATA_COLLECTION.md)** - Complete data gathering workflow
- **[Annotation Guide](docs/ANNOTATION_GUIDE.md)** *(coming soon)*
- **[Training Guide](docs/TRAINING_GUIDE.md)** *(coming soon)*
- **[API Reference](docs/API_REFERENCE.md)** *(coming soon)*

---

## 🛠️ Tools & Technologies

### Data Collection (Phase 1) ✅
- **pycocotools** - COCO dataset API
- **OpenCV** - Video processing and frame extraction
- **scikit-learn** - K-means clustering for frame selection
- **imagehash** - Perceptual hashing for duplicate detection
- **NumPy/SciPy** - Numerical processing

### Data Annotation (Phase 2) 🚧
- **LabelImg** / **Roboflow** - Annotation tools
- **Albumentations** - Data augmentation
- **pycocotools** - Format conversion

### Model Training (Phase 3) 🚧
- **Ultralytics YOLOv8** - Object detection
- **PyTorch** - Deep learning framework
- **TensorBoard** / **W&B** - Training monitoring

### Inference (Phase 4) 🚧
- **YOLOv8** - Real-time detection
- **MiDaS** - Depth estimation
- **pyttsx3** - Text-to-speech
- **PyAudio** - Spatial audio

---

## 📈 Development Roadmap

### ✅ Phase 1: Data Collection (COMPLETE)
- [x] COCO dataset downloader
- [x] SUN RGB-D integration
- [x] Video frame extraction
- [x] Intelligent frame selection
- [x] Documentation

### 🚧 Phase 2: Data Annotation (NEXT)
- [ ] COCO to YOLO converter
- [ ] Annotation verification tools
- [ ] LabelImg setup guide
- [ ] Dataset preparation script
- [ ] Augmentation pipeline

### 📅 Phase 3: Model Training
- [ ] YOLOv8 training script (2-phase)
- [ ] Custom augmentation for indoor scenes
- [ ] Evaluation metrics
- [ ] Model export (ONNX/TensorRT)
- [ ] Training documentation

### 📅 Phase 4: Inference System
- [ ] Real-time detection module
- [ ] Depth estimation integration
- [ ] Spatial analysis and grid system
- [ ] Audio feedback with stereo panning
- [ ] Main application

### 📅 Phase 5: Deployment
- [ ] Mobile app (iOS/Android)
- [ ] Edge device optimization
- [ ] User testing
- [ ] Performance optimization

---

## 🤝 Contributing

This is a research project for blind navigation assistance. Contributions are welcome!

### Areas for Contribution
- **Data Collection**: Record and share indoor navigation videos
- **Annotation**: Help annotate images (especially stairs, doors)
- **Model Training**: Experiment with different architectures
- **Audio Feedback**: Improve spatial audio algorithms
- **Testing**: User testing with blind individuals

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📊 Current Statistics

### Phase 1 Progress

| Metric | Status |
|--------|--------|
| **Scripts Written** | 4/4 ✅ |
| **Documentation** | 1/4 (25%) 🚧 |
| **COCO Integration** | Complete ✅ |
| **SUN RGB-D Integration** | Partial (manual processing) ⚠️ |
| **Custom Data Tools** | Complete ✅ |

### Dataset Collection Status

| Source | Target | Collected | Status |
|--------|--------|-----------|--------|
| COCO | 15,000-20,000 | Ready to download | ⏳ |
| SUN RGB-D | 5,000-7,000 | Manual processing | ⚠️ |
| Custom | 2,000-3,000 | Tools ready | 🔧 |

---

## ⚠️ Important Notes

### Dataset Download Sizes
- **COCO annotations**: ~500MB
- **COCO images**: ~18GB (train) + ~1GB (val)
- **SUN RGB-D**: ~18GB compressed

**Recommendation**: Start with `--no-images` flag for COCO to get annotations first, then download images selectively.

### Hardware Requirements

**Minimum (Data Collection):**
- CPU: Dual-core 2GHz+
- RAM: 8GB
- Storage: 50GB

**Recommended (Training):**
- CPU: 8-core
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- Storage: 100GB SSD

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **COCO Dataset**: [Lin et al., 2014](https://cocodataset.org/)
- **SUN RGB-D**: [Song et al., 2015](http://rgbd.cs.princeton.edu/)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MiDaS**: [Intel ISL](https://github.com/isl-org/MiDaS)

---

## 📧 Contact

For questions, suggestions, or collaboration:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/blindspot/issues)
- **Email**: your.email@example.com

---

## 🎯 Next Steps

After completing Phase 1 data collection:

```bash
# 1. Review collected data
ls data/raw/coco/images/
ls data/raw/custom/selected/

# 2. Read annotation guide
cat docs/ANNOTATION_GUIDE.md

# 3. Set up annotation tool (LabelImg)
pip install labelImg
labelImg

# 4. Start annotating custom data
# Focus on: stairs, doors, edges, low obstacles
```

**Ready to proceed to Phase 2: Data Annotation!** 🎉

---

**Built with ❤️ for accessibility and independence**
