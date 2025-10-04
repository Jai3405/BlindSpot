# BlindSpot Data Collection Guide

## Overview

This guide explains how to collect and prepare data for training the BlindSpot obstacle detection model. The dataset combines public datasets (COCO, SUN RGB-D) with custom-collected data for blind navigation scenarios.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [COCO Dataset Collection](#coco-dataset-collection)
3. [SUN RGB-D Collection](#sun-rgb-d-collection)
4. [Custom Data Collection](#custom-data-collection)
5. [Data Organization](#data-organization)
6. [Quality Guidelines](#quality-guidelines)

---

## Dataset Overview

### Target Dataset Size
- **COCO filtered**: 15,000-20,000 images
- **SUN RGB-D**: 5,000-7,000 images
- **Custom data**: 2,000-3,000 images
- **Total**: ~22,000-30,000 images

### Target Classes

#### Indoor Obstacles (from COCO)
- `person` - People in the scene
- `chair` - Chairs and seating
- `couch` - Sofas and couches
- `bed` - Beds
- `dining_table` - Tables
- `toilet` - Toilets
- `tv` - Televisions
- `laptop` - Laptops and computers
- `potted_plant` - Plants
- `refrigerator` - Large appliances
- `book` - Books and objects

#### Critical Navigation Obstacles (Custom)
- `stairs_up` - Ascending stairs
- `stairs_down` - Descending stairs
- `door_open` - Open doorways
- `door_closed` - Closed doors
- `table_edge` - Table edges and corners
- `low_obstacle` - Low obstacles (boxes, wires)
- `furniture_leg` - Furniture legs
- `narrow_passage` - Narrow corridors
- `glass_door` - Glass doors/windows

---

## COCO Dataset Collection

### Step 1: Download COCO Data

```bash
# Navigate to project root
cd /path/to/BlindSpot

# Download COCO filtered dataset (annotations only, fast)
python data_collection/download_coco.py --no-images

# Or download with images (WARNING: large download ~18GB)
python data_collection/download_coco.py
```

### What Gets Downloaded

1. **Annotations**
   - `annotations/instances_train2017.json`
   - `annotations/instances_val2017.json`
   - `annotations/instances_train2017_filtered.json` (only target classes)
   - `annotations/instances_val2017_filtered.json`

2. **Images** (if --no-images not specified)
   - `images/train2017/` - Training images
   - `images/val2017/` - Validation images

3. **Statistics**
   - `coco_statistics.json` - Dataset statistics

### Step 2: Verify Downloaded Data

```bash
# Check statistics
cat data/raw/coco/coco_statistics.json

# Sample expected output:
# {
#   "total_images": 118287,
#   "filtered_images": 18543,
#   "images_per_class": {
#     "person": 15678,
#     "chair": 12034,
#     ...
#   }
# }
```

### Filtering Logic

The script automatically filters COCO to include only images containing **at least one** target class. This reduces dataset size while maintaining relevance for indoor navigation.

---

## SUN RGB-D Collection

SUN RGB-D provides indoor scenes with depth information, which is valuable for training depth-aware models.

### Step 1: Download SUN RGB-D

```bash
# Download toolbox and metadata (small, ~100MB)
python data_collection/download_sun_rgbd.py

# For full dataset:
# 1. Visit: http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
# 2. Download SUNRGBD.zip (~18GB)
# 3. Place in: data/raw/sun_rgbd/
# 4. Run: python data_collection/download_sun_rgbd.py --extract
```

### Step 2: Process SUN RGB-D

The SUN RGB-D dataset requires manual processing due to its complex MATLAB format. Follow the generated `PROCESSING_GUIDE.md`:

```bash
cat data/raw/sun_rgbd/PROCESSING_GUIDE.md
```

**Key steps:**
1. Load metadata using scipy or MATLAB
2. Filter for target indoor scene types (bedroom, kitchen, etc.)
3. Copy RGB images and depth maps
4. Convert existing annotations to YOLO format

### Alternative: Quick Start Subset

For faster setup, manually select 500-1000 representative scenes:

```bash
# Create subset directory
mkdir -p data/raw/sun_rgbd/subset/{images,depth,annotations}

# Manually copy 500-1000 images focusing on:
# - Varied lighting conditions
# - Different room types
# - Clear obstacle visibility
# - Stairs, doors, furniture
```

---

## Custom Data Collection

Custom data is **critical** for classes not well-represented in public datasets (stairs, door states, low obstacles).

### Equipment Needed

1. **Camera**
   - Smartphone camera (1080p or higher)
   - GoPro or action camera
   - Webcam (720p minimum)

2. **Mounting** (for first-person perspective)
   - Head mount (simulates user's view)
   - Chest mount
   - Handheld (less preferred)

### Recording Guidelines

#### 1. Location Diversity

Record in multiple indoor locations:

| Location Type | Scenes to Capture | Priority |
|---------------|-------------------|----------|
| **Residential** | Living room, bedroom, kitchen, bathroom, hallway | High |
| **Stairs** | Ascending, descending, different angles | **Critical** |
| **Doorways** | Open, closed, narrow, wide | High |
| **Office** | Desks, chairs, computers, corridors | Medium |
| **Public Indoor** | Libraries, stores, lobbies | Low |

#### 2. Lighting Conditions

Capture in various lighting:
- ☀️ **Bright daylight** (natural lighting)
- 💡 **Artificial lighting** (LED, fluorescent)
- 🌙 **Dim lighting** (evening, minimal lights)
- 🔦 **Backlit scenes** (windows behind obstacles)
- 🌓 **Mixed lighting** (natural + artificial)

#### 3. Recording Technique

```bash
# Example recording session plan
Session 1: Living Room (5-10 minutes)
- Pan across room (slow, steady)
- Walk toward furniture
- Focus on table edges, chair legs
- Include both close (0.5m) and far (5m) obstacles

Session 2: Stairs (5-10 minutes) [CRITICAL]
- Record from top looking down
- Record from bottom looking up
- Walk down stairs (slowly!)
- Walk up stairs
- Different lighting on each stair set

Session 3: Doorways (3-5 minutes)
- Open doors: walk through
- Closed doors: approach
- Narrow passages
- Glass doors
```

#### 4. Video Recording Settings

- **Resolution**: 1080p (1920x1080) or higher
- **Frame rate**: 30 FPS or 60 FPS
- **Format**: MP4 (H.264) recommended
- **Duration**: 30-60 minutes total per environment type

### Step 1: Record Videos

```bash
# Create directory for raw videos
mkdir -p data/raw/custom/videos

# Record and save videos:
# - living_room_bright.mp4
# - stairs_descending.mp4
# - kitchen_dim_lighting.mp4
# etc.
```

### Step 2: Extract Frames

```bash
# Extract frames from single video
python data_collection/video_to_frames.py \
    --video data/raw/custom/videos/living_room.mp4 \
    --output data/raw/custom/frames \
    --fps 2.0 \
    --blur-threshold 100

# Batch process all videos
python data_collection/video_to_frames.py \
    --video-dir data/raw/custom/videos \
    --output data/raw/custom/frames \
    --fps 2.0
```

**Parameters:**
- `--fps 2.0`: Extract 2 frames per second (adjustable)
- `--blur-threshold 100`: Reject blurry frames (higher = stricter)
- `--min-brightness 20`: Reject too dark frames
- `--max-brightness 235`: Reject overexposed frames

**Expected output:**
- 500-1000 frames per 5-minute video
- Quality-filtered (sharp, well-lit frames)
- Statistics file with rejection reasons

### Step 3: Select Diverse Frames

```bash
# Select 300 diverse frames for annotation
python data_collection/frame_selector.py \
    --input data/raw/custom/frames \
    --output data/raw/custom/selected \
    --num-frames 300 \
    --similarity-threshold 5
```

**How it works:**
1. **Removes duplicates** using perceptual hashing
2. **Extracts visual features** (color, texture, edges)
3. **Clusters frames** into groups (K-means)
4. **Selects representative frame** from each cluster

**Result:** 300 diverse frames covering:
- Different lighting conditions
- Various viewing angles
- Range of distances
- Multiple scene types

### Step 4: Review Selected Frames

```bash
# Check selected frames
ls data/raw/custom/selected/

# Review statistics
cat data/raw/custom/selected/selection_stats.json
```

---

## Data Organization

After collection, your directory structure should look like:

```
data/
├── raw/
│   ├── coco/
│   │   ├── images/
│   │   │   ├── train2017/  (~18,000 images)
│   │   │   └── val2017/    (~2,000 images)
│   │   ├── annotations/
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── instances_train2017_filtered.json
│   │   │   └── instances_val2017_filtered.json
│   │   └── coco_statistics.json
│   │
│   ├── sun_rgbd/
│   │   ├── images/       (~5,000 images)
│   │   ├── depth/        (~5,000 depth maps)
│   │   ├── annotations/  (YOLO format)
│   │   └── sun_rgbd_statistics.json
│   │
│   └── custom/
│       ├── videos/       (raw videos)
│       ├── frames/       (all extracted frames)
│       ├── selected/     (300-500 frames for annotation)
│       └── *_stats.json  (statistics)
│
└── processed/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## Quality Guidelines

### Image Quality Requirements

✅ **Good Images:**
- Sharp and in-focus
- Adequate lighting (not too dark/bright)
- Clear obstacle visibility
- Realistic indoor scenes
- Natural camera angles

❌ **Bad Images:**
- Blurry or motion-blurred
- Overexposed or too dark
- Artificial/staged scenes
- Extreme angles
- Low resolution (<640px)

### Diversity Checklist

Ensure your dataset includes:

- [ ] Multiple room types (living room, bedroom, kitchen, bathroom, hallway)
- [ ] **Stairs in both directions** (critical)
- [ ] Open and closed doors
- [ ] Various lighting conditions (bright, dim, mixed)
- [ ] Different times of day
- [ ] Close (0.5m) and far (5m) obstacles
- [ ] Cluttered and clean scenes
- [ ] Furniture at various heights
- [ ] People in various positions
- [ ] Empty scenes (for background)

### Special Focus: Critical Classes

These classes are rare but **essential** for blind navigation:

1. **Stairs** (100+ images minimum)
   - Ascending stairs from bottom
   - Descending stairs from top
   - Side views
   - Different lighting

2. **Door States** (50+ images each)
   - Clearly open doorways
   - Closed doors
   - Partially open

3. **Low Obstacles** (50+ images)
   - Boxes on floor
   - Wires and cables
   - Small furniture
   - Pet bowls, shoes

4. **Edges** (50+ images)
   - Table corners
   - Counter edges
   - Sharp furniture edges

---

## Next Steps

After collecting data:

1. **Annotate custom data** (see `ANNOTATION_GUIDE.md`)
2. **Convert annotations** to YOLO format
3. **Prepare dataset** for training
4. **Verify data quality**

```bash
# Next: Annotation phase
# See docs/ANNOTATION_GUIDE.md
```

---

## Troubleshooting

### Issue: COCO download is slow
**Solution:** Use `--no-images` flag to download annotations only, then download images separately with a download manager.

### Issue: Not enough frames extracted
**Solution:** Lower `--blur-threshold` or adjust brightness thresholds to accept more frames.

### Issue: Selected frames are too similar
**Solution:** Increase `--similarity-threshold` in frame selector.

### Issue: SUN RGB-D is too large
**Solution:** Process a subset (500-1000 scenes) focusing on target indoor types.

---

## Resources

- [COCO Dataset](https://cocodataset.org/)
- [SUN RGB-D Dataset](http://rgbd.cs.princeton.edu/)
- [LabelImg (Annotation Tool)](https://github.com/heartexlabs/labelImg)
- [Roboflow (Annotation Platform)](https://roboflow.com/)

---

**Previous:** [README.md](../README.md) | **Next:** [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
