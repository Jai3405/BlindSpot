# COCO Image Download Status

**Started:** October 3, 2025 @ 9:45 PM
**Expected Completion:** October 4, 2025 @ 1:00-3:00 PM (15-18 hours)

---

## Current Progress

- **Total Images:** 37,937
- **Downloaded:** 9 (as of 9:47 PM)
- **Remaining:** 37,928
- **Speed:** ~1.5 seconds per image
- **Estimated Time:** 15-18 hours

---

## Download Process Details

**What's Running:**
- Process: Python script downloading COCO train2017 images
- Log File: `coco_images_download.log`
- Output Directory: `data/raw/coco/images/train2017/`

**What's Being Downloaded:**
- 37,937 filtered images containing indoor obstacles
- Classes: chair, couch, bed, dining table, tv, laptop, etc.
- Total Size: ~18GB

---

## How to Monitor Progress

### Check if still running
```bash
ps aux | grep download_coco | grep -v grep
```

### Watch live progress
```bash
tail -f coco_images_download.log
```

### Count downloaded images
```bash
ls data/raw/coco/images/train2017/ | wc -l
```

### Check disk space used
```bash
du -sh data/raw/coco/images/train2017/
```

---

## Progress Checkpoints

| Time | Expected Images | Check Command |
|------|----------------|---------------|
| **11:00 PM** | ~2,400 | `ls data/raw/coco/images/train2017/ \| wc -l` |
| **2:00 AM** | ~7,200 | Same |
| **6:00 AM** | ~16,800 | Same |
| **10:00 AM** | ~26,400 | Same |
| **2:00 PM** | ~37,937 (Complete!) | Same |

---

## If Download Stops/Fails

### Check if process died
```bash
ps aux | grep download_coco | grep -v grep
```

If no output, process stopped. Check log for errors:
```bash
tail -50 coco_images_download.log
```

### Resume download
The script automatically skips already-downloaded images:
```bash
source venv/bin/activate
python -c "
from data_collection.download_coco import COCODownloader
import json

downloader = COCODownloader('data/raw/coco')

# Load filtered IDs
with open('data/raw/coco/annotations/instances_train2017_filtered.json', 'r') as f:
    data = json.load(f)
    image_ids = set([img['id'] for img in data['images']])

# Resume download (auto-skips existing)
downloader.download_images('train2017', image_ids)
"
```

---

## Expected Final Results

When complete, you'll have:

```
data/raw/coco/
├── annotations/
│   ├── instances_train2017.json (448MB)
│   └── instances_train2017_filtered.json (153MB)
├── images/
│   └── train2017/ (37,937 images, ~18GB)
└── coco_statistics.json
```

---

## Next Steps (When Download Completes)

1. **Verify download completed:**
   ```bash
   ls data/raw/coco/images/train2017/ | wc -l
   # Should show: 37937
   ```

2. **Check final size:**
   ```bash
   du -sh data/raw/coco/images/train2017/
   # Should show: ~18GB
   ```

3. **Review download log for errors:**
   ```bash
   grep -i "failed\|error" coco_images_download.log
   ```

4. **Ready for Phase 2!**
   - COCO → YOLO annotation converter
   - Dataset preparation
   - Training setup

---

## Troubleshooting

### Download is very slow
- **Normal:** COCO servers rate-limit downloads
- **Speed:** 1-2 seconds per image is expected
- **Alternative:** Use COCO's official bulk download (requires registration)

### Process killed / Out of memory
- **Solution:** Restart with same command (auto-resumes)
- Check available RAM: `free -h` (Linux) or `top` (Mac)

### Disk space full
- Check space: `df -h`
- Need ~20GB free (18GB images + overhead)
- Clear space and restart download

### Network interruption
- **Solution:** Just restart - script skips existing images
- Already downloaded images won't re-download

---

## Stats to Track

Create a simple tracking file:

```bash
# Run this every few hours to track progress
echo "$(date): $(ls data/raw/coco/images/train2017/ | wc -l) images" >> download_progress.txt
```

Then view progress:
```bash
cat download_progress.txt
```

---

## When to Check

**Before Bed:**
- Verify process is running
- Check ~100-200 images downloaded

**Morning (8 AM):**
- Should have ~19,000 images
- Check log for errors

**Afternoon (2 PM):**
- Should be complete or near complete
- Verify final count

---

## Contact / Resume

Once download completes (or if you need help):

1. Verify completion: `ls data/raw/coco/images/train2017/ | wc -l`
2. Check for errors: `tail -100 coco_images_download.log`
3. Ready to proceed to **Phase 2: Data Annotation**

---

**Status:** 🟢 **DOWNLOADING IN BACKGROUND**

Download will continue even if you:
- Close this terminal
- Put computer to sleep (will pause and resume)
- Disconnect from network briefly (will retry)

**Important:** Keep computer powered on and connected to internet!

---

*Last Updated: Oct 3, 2025 @ 9:47 PM*
*Expected Completion: Oct 4, 2025 @ 1:00-3:00 PM*
