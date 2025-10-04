# BlindSpot Quick Start Guide

**Phase 1: Data Collection - Get Started in 5 Minutes**

---

## ⚡ Express Setup

```bash
# 1. Setup environment (one-time)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 2. Activate environment
source venv/bin/activate

# 3. You're ready!
```

---

## 🎯 Common Tasks

### Download COCO Dataset

**Option A: Annotations Only (Fast - 500MB)**
```bash
python data_collection/download_coco.py --no-images
```
⏱️ ~5-10 minutes | 💾 500MB

**Option B: Full Dataset (Slow - 18GB)**
```bash
python data_collection/download_coco.py
```
⏱️ ~2-4 hours | 💾 18GB

**Expected Result:**
- ✅ 15,000-20,000 filtered images
- ✅ Annotations in COCO format
- ✅ Statistics file

---

### Process Your Own Videos

**Step 1: Extract Frames**
```bash
python data_collection/video_to_frames.py \
    --video my_indoor_video.mp4 \
    --output data/raw/custom/frames \
    --fps 2.0
```

**What this does:**
- Extracts 2 frames per second
- Removes blurry frames
- Filters out too dark/bright frames
- Saves to organized folders

⏱️ ~2-5 minutes per video | Output: 500-1000 frames per 5-min video

**Step 2: Select Best Frames**
```bash
python data_collection/frame_selector.py \
    --input data/raw/custom/frames \
    --output data/raw/custom/selected \
    --num-frames 300
```

**What this does:**
- Removes duplicate frames
- Clusters by visual similarity
- Selects 300 diverse frames
- Perfect for annotation

⏱️ ~5-10 minutes | Output: 300 ready-to-annotate frames

---

### Batch Process Multiple Videos

```bash
# Put all videos in a folder
mkdir -p videos/

# Extract frames from all videos
python data_collection/video_to_frames.py \
    --video-dir videos/ \
    --output data/raw/custom/frames

# Select diverse frames
python data_collection/frame_selector.py \
    --input data/raw/custom/frames \
    --output data/raw/custom/selected \
    --num-frames 500
```

---

## 📊 Check Your Progress

```bash
# View COCO statistics
cat data/raw/coco/coco_statistics.json

# Count extracted frames
ls data/raw/custom/frames/*/ | wc -l

# View selected frames
ls data/raw/custom/selected/
```

---

## 🎬 Video Recording Tips

### Where to Record
- ✅ Your home (living room, kitchen, bedroom)
- ✅ Stairs (CRITICAL - record both directions)
- ✅ Doorways (open and closed)
- ✅ Hallways and narrow spaces

### How to Record
1. **Use smartphone** (1080p, 30fps)
2. **Mount at chest/head level** (user perspective)
3. **Move slowly** (2-3 seconds per obstacle)
4. **Vary lighting** (bright, dim, mixed)
5. **Record 5-10 minutes per location**

### Priority Footage
Focus on these **critical** classes:
- 🔥 Stairs (up & down) - 10+ minutes
- 🚪 Doors (open/closed) - 5 minutes
- 📦 Low obstacles (boxes, wires) - 5 minutes
- 🪑 Furniture edges - 5 minutes

---

## 🛠️ Troubleshooting

### "No images found"
**Solution:** Check that video file path is correct and file is readable.

### "Too few frames extracted"
**Solution:** Lower blur threshold:
```bash
python data_collection/video_to_frames.py \
    --video my_video.mp4 \
    --blur-threshold 50  # Lower = more frames
```

### "Selected frames are too similar"
**Solution:** Increase similarity threshold:
```bash
python data_collection/frame_selector.py \
    --input data/raw/custom/frames \
    --similarity-threshold 10  # Higher = less strict
```

### "COCO download is slow"
**Solution:** Download annotations first, then download images separately with a download manager.

---

## 📈 Progress Checklist

### Phase 1: Data Collection

- [ ] Environment setup complete
- [ ] COCO annotations downloaded
- [ ] COCO images downloaded (optional)
- [ ] Recorded 30+ minutes of indoor videos
- [ ] Extracted frames from videos
- [ ] Selected 300-500 diverse frames
- [ ] Focused on stairs (CRITICAL)
- [ ] Recorded various lighting conditions

**Target:** 300-500 custom frames ready for annotation

---

## ➡️ Next Steps

Once you have collected data:

1. **Review frames**: Check that selected frames are diverse and high-quality
   ```bash
   open data/raw/custom/selected/
   ```

2. **Read annotation guide**: Learn how to annotate your frames
   ```bash
   cat docs/DATA_COLLECTION.md
   ```

3. **Setup annotation tool**: Install LabelImg
   ```bash
   pip install labelImg
   labelImg
   ```

4. **Start annotating**: Focus on critical classes first (stairs, doors)

5. **Move to Phase 2**: Annotation pipeline (coming soon)

---

## 🎯 Goals Summary

By end of Phase 1, you should have:
- ✅ 15,000-20,000 COCO images (optional: annotations only)
- ✅ 300-500 custom frames (selected & quality-filtered)
- ✅ Focus on critical classes (stairs, doors, edges)
- ✅ Variety in lighting and environments

**Ready to annotate!** 🎉

---

## 📚 More Resources

- **Full Documentation**: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
- **Project README**: [README.md](README.md)
- **Script Help**:
  ```bash
  python data_collection/download_coco.py --help
  python data_collection/video_to_frames.py --help
  python data_collection/frame_selector.py --help
  ```

---

## 💬 Need Help?

- 📖 Read: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
- 🐛 Issues: Create a GitHub issue
- 💡 Questions: Open a discussion

**Happy collecting!** 🦯✨
