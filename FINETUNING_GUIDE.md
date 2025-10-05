# Fine-Tuning Guide: Indoor Obstacle Detection

## 🎯 Goal
Improve BlindSpot AI by adding indoor obstacle detection (walls, doors, stairs, furniture) to the existing COCO-trained model.

## 📋 Step-by-Step Process

### Step 1: Download Roboflow Datasets

**Option A - Manual Download (Recommended):**

1. **Dataset 1: Indoor Obstacle Detection**
   - Visit: https://universe.roboflow.com/indooroutdoornavigation/indoor-obstacle-detection
   - Click "Download Dataset"
   - Format: Select "YOLOv8"
   - Download and extract to: `data/raw/roboflow/dataset1/`

2. **Dataset 2: Indoor Obstacles**
   - Visit: https://universe.roboflow.com/project-wcqez/indoor-obstacles
   - Click "Download Dataset"  
   - Format: Select "YOLOv8"
   - Download and extract to: `data/raw/roboflow/dataset2/`

**Option B - With API Key:**
```bash
export ROBOFLOW_API_KEY='your_api_key'
venv/bin/python data_collection/download_roboflow_datasets.py
```

### Step 2: Fine-Tune the Model

Once datasets are downloaded:

```bash
venv/bin/python model_training/finetune_indoor_obstacles.py
```

**What happens:**
- Loads your COCO-trained model (46.7% mAP)
- Freezes early layers (keeps COCO knowledge)
- Trains on indoor obstacles (walls, doors, stairs, furniture)
- Training time: ~20-30 minutes on M2 Pro
- Saves to: `runs/finetune/indoor_obstacles/weights/best.pt`

### Step 3: Test the Fine-Tuned Model

```bash
./venv/bin/python demo_blindspot.py \
  --mode webcam \
  --model runs/finetune/indoor_obstacles/weights/best.pt
```

## 🎁 Expected Improvements

**Before (COCO only):**
- ✅ Detects: person, chair, couch, laptop, car
- ❌ Misses: walls, doors, stairs, cabinets

**After (COCO + Indoor):**
- ✅ Detects: ALL above + walls, doors, stairs, furniture
- ✅ Better indoor navigation
- ✅ Earlier obstacle warnings
- ✅ Safer navigation

## 📊 Transfer Learning Benefits

1. **Faster Training:** 30 min vs 5 hours (from scratch)
2. **Better Performance:** Combines COCO + Indoor knowledge
3. **Wider Coverage:** Indoor + outdoor objects
4. **No Data Loss:** Keeps existing COCO detections

## 🔄 Quick Summary

```bash
# 1. Download datasets (5-10 min)
#    - Indoor Obstacle Detection → dataset1
#    - Indoor Obstacles → dataset2

# 2. Fine-tune (20-30 min)
venv/bin/python model_training/finetune_indoor_obstacles.py

# 3. Test (immediate)
./venv/bin/python demo_blindspot.py \
  --mode webcam \
  --model runs/finetune/indoor_obstacles/weights/best.pt
```

## 📈 Next Steps After Fine-Tuning

If detection is still not perfect:
1. Record custom videos of your environment
2. Label 50-100 frames with LabelImg
3. Fine-tune again on custom data
4. Iterate until satisfied

The model will get progressively better with each fine-tuning iteration!
