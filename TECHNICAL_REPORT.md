# Technical Implementation Report: BlindSpot AI

**Author**: Jai Sungkur
**Date**: October 6, 2025
**Version**: 2.1

---

## Abstract

This report provides a detailed technical breakdown of the implementation of the BlindSpot AI project, a real-time navigation assistant for the visually impaired. It focuses on the practical execution of the data pipeline, model training, and the final inference application. The project's success hinges on a sophisticated data strategy that programmatically collects, preprocesses, and merges disparate datasets (COCO, Roboflow Indoor). This document details the algorithms used for intelligent data selection, annotation format conversion, and the creation of a unified 24-class dataset. It further analyzes the specifics of the 2-phase transfer learning strategy employed on a YOLOv8n architecture and the architecture of the final `demo_blindspot.py` application. Finally, it documents the challenges encountered during training and the specific interventions taken to ensure a successful outcome, culminating in a final model with a peak mAP of 0.290.

---

## 1. Data Acquisition and Collection

The foundation of this project was the creation of a high-quality, diverse dataset. This was not a manual process but a programmatic pipeline designed for scalability and reproducibility, orchestrated by scripts in the `/data_collection/` directory.

### 1.1. COCO Dataset Integration

-   **Script**: `download_coco.py`
-   **Technical Implementation**: The script leverages the `pycocotools` API to interact with the COCO dataset. Instead of downloading the entire 120k+ image dataset, it first fetches the annotation files. It then programmatically filters these annotations to build a list of image IDs that contain at least one of the 17 pre-selected indoor classes. This targeted approach reduces the required download size from over 20GB to a more manageable subset, ensuring relevance and efficiency.

### 1.2. Custom Video Data Pipeline

To capture critical navigational classes not present in academic datasets, a pipeline was developed to process custom-recorded videos.

1.  **Frame Extraction**: The `video_to_frames.py` script uses OpenCV (`cv2.VideoCapture`) to iterate through video files. It implements a frame-rate-limiting logic to extract frames at a specified interval (e.g., 2 FPS), preventing the creation of thousands of nearly identical, redundant images.

2.  **Intelligent Frame Selection**: A simple frame dump results in a low-quality dataset. The `frame_selector.py` script implements a sophisticated unsupervised learning approach to select a diverse and representative subset of frames for annotation:
    -   **Perceptual Hashing**: Initially, the `imagehash` library is used to calculate a perceptual hash for each frame. Frames with a hash distance below a certain threshold are discarded as near-duplicates.
    -   **Feature Extraction**: For the remaining unique frames, a feature vector is extracted using a combination of color histograms and edge detection histograms.
    -   **K-Means Clustering**: The feature vectors are then clustered using `sklearn.cluster.KMeans`. The number of clusters (`k`) is chosen to be the desired number of final frames.
    -   **Centroid Selection**: The script then selects the single frame closest to the centroid of each cluster. This ensures the final selection of frames is maximally diverse.

---

## 2. Data Annotation and Preprocessing

With raw image data collected, the next critical phase was to standardize annotations and structure the data for the YOLOv8 trainer.

### 2.1. Annotation Conversion

-   **Script**: `coco_to_yolo.py`
-   **Technical Implementation**: This script parses the complex COCO JSON format. For each image, it finds the corresponding annotations, extracts the `class_id` and `bbox` (`[x, y, width, height]`), and converts the bounding box coordinates from the absolute pixel format of COCO to the relative format required by YOLO (i.e., `x_center`, `y_center`, `width`, `height`, all scaled to be between 0 and 1). It then writes a new `.txt` file for each image.

### 2.2. Dataset Merging and Standardization

-   **Script**: `data_preparation/merge_datasets.py`
-   **Technical Implementation**: This script performs the final, critical preprocessing step.
    1.  **Unified Class Mapping**: It defines a master list of 24 classes.
    2.  **File System Unification**: It creates a new, clean directory structure (`/data/processed/merged_navigation/`).
    3.  **Programmatic Rewriting of Annotations**: The script iterates through the pre-converted YOLO annotation files from each source dataset. For each file, it reads every line, remaps the original `class_id` to its new, unified ID (e.g., COCO class `2` might remain `2`, while Indoor class `0` becomes `17`), and writes a new, corrected annotation file.
    4.  **Filename Prefixing**: To prevent name collisions, all image and annotation files are programmatically renamed with a prefix (`coco_`, `indoor_`) during the copy process.
    5.  **YAML Generation**: Finally, the script auto-generates the `dataset.yaml` file, writing the correct paths and the full list of 24 class names in the correct order.

---

## 3. Model Training: A Deep Dive

The training was executed via the `train_merged_navigation.py` script, which leverages the `ultralytics` Python library.

### 3.1. Data Augmentation: The Technical Details

Data augmentation was a carefully configured pipeline that operates on the CPU, preparing batches of transformed data while the GPU is busy with training.

-   **Mosaic Augmentation (`mosaic=0.3`)**: Stitches four random images together, forcing the model to learn object detection without strong reliance on typical image context and improving detection of partially occluded objects.
-   **Geometric Augmentations (`degrees`, `translate`, `scale`, `fliplr`)**: Applied as a single, combined **affine transformation**, teaching the model rotational and scale invariance.
-   **Color Space Augmentations (`hsv_h`, `hsv_s`, `hsv_v`)**: Applied in the **Hue-Saturation-Value (HSV)** color space to simulate real-world lighting changes.

### 3.2. The 2-Phase Training: Implementation Details

-   **Phase 1 (`freeze=10`)**: The `freeze` parameter freezes the first 10 modules of the YOLOv8 backbone (low-level feature extractors like `Conv2d` and `C2f`). The optimizer is only passed the parameters of the unfrozen layers, allowing the model to learn new classes without corrupting its core knowledge.

-   **Phase 2 (`freeze=None`)**: All layers were unfrozen. A new optimizer was initialized with all model parameters and an extremely low learning rate (`lr0=0.00003`). A higher rate would have caused destructive gradient updates to the sensitive early layers, effectively destroying the pre-trained COCO features.

---

## 4. Final Model Evaluation

### 4.1. Quantitative Results

The final model is definitively the best model for the project's goal, successfully balancing two domains.

| Model | Peak Accuracy (mAP@.50-.95) | Key Capability |
| :--- | :--- | :--- |
| Original COCO | ~0.467 | Detects general objects, but **no navigational cues**. |
| Indoor-only | ~0.323 | Detects navigational cues, but **no general objects**. |
| **Final Phase 2 (Fine-Tuned)** | **0.290** | **Detects all 24 classes with the highest accuracy.** |

### 4.2. Key Metrics Analysis

-   **mAP@.50-.95**: **0.290** - The primary "overall score," representing a strong result for a complex 24-class model.
-   **Precision**: **~58%** - When the model makes a detection, it is correct 58% of the time, indicating a low rate of false alarms.
-   **Recall**: **~43%** - The model successfully finds 43% of all available objects in a scene, providing a solid baseline for environmental awareness.

---

## 5. Key Implementation Scripts

### 5.1. `model_training/train_merged_navigation.py`

-   **Purpose**: This is the master script that automates the entire 2-phase training workflow.
-   **Functionality**: It contains two sequential `model.train()` calls. The first call executes Phase 1 with `freeze=10`. Upon completion, the script immediately loads the `best.pt` artifact from the Phase 1 output directory and begins the second `model.train()` call with the Phase 2 parameters (`freeze=None`, lower learning rate), creating a seamless end-to-end training pipeline.

### 5.2. `inference/blindspot_engine.py`

-   **Purpose**: This script encapsulates the core inference logic, acting as a bridge between the ML models and the main application.
-   **Functionality**: The `BlindSpotEngine` class initializes and holds both the trained YOLOv8 model and the MiDaS depth estimation model. Its primary method, `process_frame`, takes a single image, runs both models in sequence, and then calls the `SpatialAnalyzer` to correlate the outputs, producing a final list of `SpatialObject` data structures that contain the class name, distance, and position for each detection.

---

## 6. The `demo_blindspot.py` Inference Application

This script serves as the user-facing application, integrating all project components into a functional prototype.

### 6.1. Architecture

The demo is built around a main `BlindSpotDemo` class which manages the application state and orchestrates the following components:

-   **`BlindSpotEngine`**: The core engine for processing frames (as described above).
-   **`AudioFeedback`**: A dedicated class that manages a text-to-speech (TTS) engine (`pyttsx3`). It contains the logic for generating natural language alerts.
-   **OpenCV (`cv2`)**: Used for all video I/O (capturing from webcam, reading video files) and for rendering the visual interface (drawing bounding boxes, text overlays).

### 6.2. Real-time Execution Flow

Inside the main `while True:` loop of the `run_webcam` method, the following occurs for each frame:

1.  **Capture**: A frame is read from the `cv2.VideoCapture` object.
2.  **Process**: The frame is sent to `self.engine.process_frame()`.
3.  **Analyze**: The engine returns a dictionary containing the detected objects, their spatial information, and navigation hints.
4.  **Prioritize**: The application determines a `current_priority` based on the closest detected object.
5.  **Adaptive Audio**: Based on the priority, an `audio_interval` is set (e.g., 5 frames for critical alerts, 30 for clear paths). Audio feedback is only triggered if `frame_count % audio_interval == 0`.
6.  **Visualize**: The results are drawn on the frame, including a dynamic, color-coded status overlay.
7.  **Display**: The final frame is shown on screen using `cv2.imshow()`.
8.  **Input Handling**: `cv2.waitKey()` listens for keyboard input to handle interactive controls (`q`, `a`, `d`, `s`, `h`).

### 6.3. Interactive Features Implementation

The interactive controls are implemented via a simple key-checking block within the main loop. For example, when the `d` key is pressed, the `_describe_scene` method is called, which takes the latest results dictionary and provides a comprehensive audio summary of all detected objects, grouped by distance.

---

## 7. Conclusion and Future Work

This project successfully implemented a complex computer vision pipeline, from programmatic data synthesis to a fully-featured inference application. The final fine-tuned model is a testament to the effectiveness of the 2-phase training strategy. The challenges faced and overcome during the training process underscore the practical realities of machine learning development and provide valuable insights for future work.

### Future Work

-   **Improve mAP**: Experiment with different YOLOv8 model sizes (e.g., YOLOv8s/m) or more advanced augmentation.
-   **Dataset Expansion**: Add more custom-annotated data for critical edge cases.
-   **Edge Deployment**: Optimize the model (e.g., via quantization) and deploy to a low-power edge device (e.g., Jetson Nano, OAK-D).
-   **Real-World Testing**: Conduct user acceptance testing with visually impaired individuals to gather feedback on the audio guidance.
