# Navigation Dataset Strategy

## Current Problem
- COCO model (46.7% mAP) misses critical obstacles:
  ❌ Walls
  ❌ Doors  
  ❌ Stairs
  ✅ Has: person, chair, couch, car (good but incomplete)

## Smart Solution: Targeted Fine-Tuning

### Option 1: Roboflow Pre-Filtered (FASTEST) ⭐
- **Size:** 200-500MB (vs 18GB SUN RGB-D)
- **Time:** 10 minutes download
- **Already in YOLO format:** No conversion needed
- **Contains:** Stairs, doors, walls, furniture

**How to get it:**
1. Visit [Roboflow Universe](https://universe.roboflow.com)
2. Search: "indoor navigation" or "stairs detection"
3. Download in YOLOv8 format
4. Merge with our COCO model

### Option 2: Manual Subset from SUN RGB-D
- Extract ONLY stairs/doors/walls images
- **Size:** ~500MB (1,500 images)
- **Time:** 1-2 hours (manual filtering)
- Download annotations → filter → download images

### Option 3: Augment with Custom Data (QUICKEST TEST)
- **Record 50-100 videos** of your environment
- **Extract frames** with obstacles
- **Label manually** (only stairs, doors, walls)
- **Size:** <100MB, perfectly tailored
- **Train:** 30 minutes

## Recommended: Option 3 (Custom Recording)

**Why it's best:**
✅ Tailored to YOUR environment  
✅ Small dataset = fast training
✅ Test immediately
✅ Can expand later

**Steps:**
1. Record 2-3 minute video walking around
2. Extract frames every 0.5 seconds (~300 frames)
3. Label obstacles using LabelImg (10-15 minutes)
4. Fine-tune existing COCO model (20 minutes)
5. Test in real-time

