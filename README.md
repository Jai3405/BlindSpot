# BlindSpot AI 🦯

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**An AI-Powered Navigation Assistant for the Visually Impaired, providing real-time environmental awareness through intelligent audio feedback.**

BlindSpot is an end-to-end computer vision system that transforms a standard webcam feed into a rich, audible description of the surrounding world. It combines state-of-the-art object detection and depth estimation to identify obstacles, calculate their distance and position, and guide the user through complex indoor environments safely.

---

## 🌟 Key Features

- **Comprehensive Indoor Object Detection**: Identifies 24 essential indoor classes, including furniture (`chair`, `table`) and critical navigation obstacles (`stairs`, `door`, `wall`).
- **Real-time Depth Estimation**: Accurately calculates the distance to detected objects using the MiDaS depth model.
- **Intelligent Audio Guidance**: Provides clear, concise audio alerts about the environment.
  - **Adaptive Frequency**: Alerts become more frequent as obstacles get closer.
  - **Spatial Awareness**: Describes object positions ("on your left", "directly ahead").
  - **Safe Path Suggestions**: Proactively suggests safe directions ("Safe path to your right").
- **Interactive In-Demo Controls**: Full control over the application without stopping.
  - Toggle audio, request detailed scene descriptions, print summaries, and access help on the fly.
- **Enhanced Visual Interface**: A color-coded overlay provides at-a-glance status, object counts, FPS, and a safe direction indicator for sighted assistance and debugging.
- **Complete 2-Phase Training Pipeline**: A sophisticated, reproducible workflow for merging datasets and fine-tuning models for specialized tasks.

---

## 🚀 Live Demo

Below is a snapshot of the BlindSpot demo in action, identifying a `chair` and a `wall` while providing real-time status information in the overlay.

*(This is a placeholder for a GIF of the demo running)*
```
STATUS: CAUTION | Objects: 2 | FPS: 15 | Audio: ON
Safe Direction: Right

[INFO] chair detected at 2.1 meters on your left.
[INFO] wall detected at 4.5 meters directly ahead.
```

---

## 🧠 The Model: Performance & Strategy

The core of BlindSpot is a custom-trained YOLOv8n model, developed using a sophisticated 2-phase transfer learning strategy to achieve a unique set of capabilities.

### Training Strategy

1.  **Dataset Merging**: We combined two datasets:
    *   **COCO**: Provided a strong foundation with 17 general object classes.
    *   **Indoor Obstacles**: A custom dataset with 7 crucial navigation-related classes (`door`, `wall`, `stairs`, etc.).
    This resulted in a comprehensive dataset of **5,301 images** across **24 classes**.

2.  **Phase 1: Frozen Backbone Training (15 Epochs)**: We first trained only the detection head of the model, keeping the main body (backbone) frozen. This allowed the model to learn the 7 new indoor classes without forgetting its powerful, pre-existing knowledge from COCO.

3.  **Phase 2: Full Fine-Tuning (25 Epochs)**: We then unfroze the entire model and continued training at a very low learning rate. This allowed the whole network to gently adapt and optimize its performance across all 24 classes simultaneously.

### Final Model Performance

This 2-phase approach was a success, creating a single, versatile model that outperforms specialized models for this task.

| Model | Peak Accuracy (mAP@.50-.95) | Can Detect Furniture, etc.? | Can Detect Walls, Doors, etc.? |
| :--- | :--- | :--- | :--- |
| Original COCO | ~0.467 | ✅ Yes | ❌ **No** |
| Indoor-only | ~0.323 | ❌ **No** | ✅ Yes |
| **Our Final Model** | **0.290** | ✅ **Yes** | ✅ **Yes** |

The final model's key metrics on the merged validation set are:
- **mAP@.50-.95**: **0.290**
- **Precision**: **~0.58**
- **Recall**: **~0.43**

---

## 🛠️ Tech Stack

- **Core Framework**: Python 3.10, PyTorch
- **Object Detection**: Ultralytics YOLOv8
- **Depth Estimation**: MiDaS (v3.1)
- **Audio**: pyttsx3 (Text-to-Speech)
- **Core Libraries**: OpenCV, NumPy, SciPy

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jai3405/BlindSpot.git
    cd BlindSpot
    ```

2.  **Run the setup script:**
    This will create a Python virtual environment and install all required dependencies.
    ```bash
    chmod +x scripts/setup_environment.sh
    ./scripts/setup_environment.sh
    ```

3.  **Activate the environment:**
    You must activate the virtual environment each time you want to run the application.
    ```bash
    source venv/bin/activate
    ```

---

## ▶️ Usage

The primary application is `demo_blindspot.py`. It can be run in several modes.

### Live Webcam Demo

This is the main mode of operation. Make sure your webcam is connected.
```bash
python demo_blindspot.py --mode webcam --camera 0 --model runs/merged_retrain/phase2_unfrozen/weights/best.pt
```
*   `--camera`: Use `0` for a built-in laptop webcam, `1` for an external USB webcam, etc.

### Video File Demo

Analyze a pre-recorded video file.
```bash
python demo_blindspot.py --mode video --source /path/to/your/video.mp4 --model runs/merged_retrain/phase2_unfrozen/weights/best.pt
```

### In-Demo Interactive Controls

- **`q`**: Quit the application.
- **`a`**: Toggle audio feedback ON/OFF.
- **`d`**: Request a detailed, one-time description of the current scene.
- **`s`**: Print a text summary of detected objects to the console.
- **`h`**: Display the help menu with all available controls.

---

## 🔄 Re-training the Model

The full training pipeline is included. You can re-run the entire 2-phase training process using the following command:
```bash
python model_training/train_merged_navigation.py
```
**Note**: This is a very long and computationally expensive process.

---

## 🙏 Acknowledgments

This project stands on the shoulders of giants. Our sincere thanks to the creators and maintainers of:
- **COCO Dataset**: [Lin et al., 2014](https://cocodataset.org/)
- **Roboflow**: For providing the open-source Indoor Obstacles dataset.
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MiDaS**: [Intel ISL](https://github.com/isl-org/MiDaS)

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.