# Continuity Camera Setup for BlindSpot AI

## Step-by-Step Setup

### 1. System Requirements
- ✅ macOS Ventura or later
- ✅ iPhone with iOS 16 or later
- ✅ Same Apple ID on both devices

### 2. Enable Continuity Camera

**On iPhone:**
1. Open Settings → General → AirPlay & Handoff
2. Enable "Continuity Camera"

**On Mac:**
1. Open System Settings → General → AirDrop & Handoff
2. Enable "iPhone Wireless Camera"

### 3. Connection Setup
1. Make sure iPhone and Mac are:
   - Signed in to same Apple ID
   - On same WiFi network
   - Bluetooth enabled on both
   - iPhone is unlocked and nearby (within ~30 feet)

2. iPhone doesn't need any app open - it will activate automatically

### 4. Test the Connection

Run this command:
```bash
python demo_blindspot.py --mode webcam --camera 0
```

When prompted, select your iPhone as the camera source.

### 5. Alternative Camera IDs

If camera 0 doesn't work, try:
```bash
# Try camera 1
python demo_blindspot.py --mode webcam --camera 1

# Try camera 2
python demo_blindspot.py --mode webcam --camera 2
```

## Troubleshooting

If iPhone doesn't appear:
1. Restart both devices
2. Toggle WiFi/Bluetooth off and on
3. Sign out and back into iCloud on both
4. Make sure Handoff is enabled on both devices
5. Update to latest iOS/macOS versions

## Once Connected

Controls:
- Press 'q' to quit
- Press 's' for object summary
- Press 'a' to toggle audio feedback

The system will show:
- Real-time object detection
- Depth estimation
- Distance measurements
- Navigation hints
- Audio alerts
