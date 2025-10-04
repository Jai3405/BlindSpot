#!/usr/bin/env python3
"""
Intelligent Frame Selector for BlindSpot
Selects diverse, high-quality frames for annotation using clustering
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import json
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import imagehash
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameSelector:
    """Select diverse frames for annotation using intelligent sampling"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        num_frames: int = 300,
        similarity_threshold: int = 5
    ):
        """
        Initialize frame selector

        Args:
            input_dir: Directory containing extracted frames
            output_dir: Directory to save selected frames
            num_frames: Number of frames to select
            similarity_threshold: Perceptual hash distance threshold
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        self.similarity_threshold = similarity_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_frames': 0,
            'removed_duplicates': 0,
            'selected_frames': 0,
            'clusters': 0
        }

    def find_images(self) -> List[Path]:
        """Find all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        images = []

        for ext in image_extensions:
            images.extend(self.input_dir.rglob(f'*{ext}'))
            images.extend(self.input_dir.rglob(f'*{ext.upper()}'))

        return sorted(images)

    def compute_perceptual_hash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute perceptual hash for duplicate detection

        Args:
            image_path: Path to image

        Returns:
            Perceptual hash
        """
        image = Image.open(image_path)
        return imagehash.phash(image)

    def remove_duplicates(self, image_paths: List[Path]) -> List[Path]:
        """
        Remove duplicate/similar images using perceptual hashing

        Args:
            image_paths: List of image paths

        Returns:
            List of unique image paths
        """
        logger.info("Removing duplicate frames...")

        unique_images = []
        seen_hashes = []

        for img_path in tqdm(image_paths, desc="Checking duplicates"):
            img_hash = self.compute_perceptual_hash(img_path)

            # Check if similar to any seen hash
            is_duplicate = False
            for seen_hash in seen_hashes:
                if img_hash - seen_hash <= self.similarity_threshold:
                    is_duplicate = True
                    self.stats['removed_duplicates'] += 1
                    break

            if not is_duplicate:
                unique_images.append(img_path)
                seen_hashes.append(img_hash)

        logger.info(f"✓ Removed {self.stats['removed_duplicates']} duplicate frames")
        logger.info(f"  Unique frames: {len(unique_images)}")

        return unique_images

    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract features from image for clustering

        Features include:
        - Color histogram (RGB)
        - Brightness statistics
        - Edge density
        - Texture features

        Args:
            image_path: Path to image

        Returns:
            Feature vector
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return np.zeros(128)  # Return zero vector if image can't be read

        # Resize for consistent processing
        image = cv2.resize(image, (224, 224))

        features = []

        # 1. Color histogram (48 features: 16 bins per channel)
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [16], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)

        # 2. Brightness statistics (4 features)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])

        # 3. Edge density (1 feature)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # 4. Texture features using LBP-like approach (8 features)
        # Simplified texture measure using gradient directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # Histogram of gradient directions (8 bins)
        hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)

        return np.array(features)

    def cluster_frames(self, image_paths: List[Path]) -> Tuple[np.ndarray, KMeans]:
        """
        Cluster frames based on visual features

        Args:
            image_paths: List of image paths

        Returns:
            (cluster_labels, kmeans_model)
        """
        logger.info("Extracting features from frames...")

        # Extract features from all images
        features = []
        valid_paths = []

        for img_path in tqdm(image_paths, desc="Extracting features"):
            feat = self.extract_features(img_path)
            if np.any(feat):  # Only include valid features
                features.append(feat)
                valid_paths.append(img_path)

        features = np.array(features)

        # Determine number of clusters
        n_clusters = min(self.num_frames, len(valid_paths))
        self.stats['clusters'] = n_clusters

        logger.info(f"\nClustering {len(valid_paths)} frames into {n_clusters} clusters...")

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(features_scaled)

        logger.info(f"✓ Clustering complete")

        return labels, kmeans, valid_paths, features_scaled

    def select_representative_frames(
        self,
        image_paths: List[Path],
        labels: np.ndarray,
        features: np.ndarray,
        kmeans: KMeans
    ) -> List[Path]:
        """
        Select one representative frame from each cluster

        Selects the frame closest to cluster center

        Args:
            image_paths: List of image paths
            labels: Cluster labels
            features: Feature vectors
            kmeans: Trained KMeans model

        Returns:
            List of selected image paths
        """
        logger.info("Selecting representative frames from each cluster...")

        selected_frames = []

        for cluster_id in tqdm(range(kmeans.n_clusters), desc="Selecting frames"):
            # Get all frames in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Get cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]

            # Find frame closest to center
            cluster_features = features[cluster_indices]
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]

            selected_frames.append(image_paths[closest_idx])

        self.stats['selected_frames'] = len(selected_frames)

        logger.info(f"✓ Selected {len(selected_frames)} representative frames")

        return selected_frames

    def save_selected_frames(self, selected_frames: List[Path]):
        """
        Copy selected frames to output directory

        Args:
            selected_frames: List of selected frame paths
        """
        logger.info(f"Copying selected frames to {self.output_dir}...")

        for i, frame_path in enumerate(tqdm(selected_frames, desc="Copying frames")):
            # Create descriptive filename
            output_name = f"selected_{i:04d}_{frame_path.name}"
            output_path = self.output_dir / output_name

            shutil.copy2(frame_path, output_path)

        logger.info(f"✓ Frames saved to {self.output_dir}")

    def generate_statistics(self):
        """Generate selection statistics"""
        stats_file = self.output_dir / 'selection_stats.json'

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("Frame Selection Statistics")
        logger.info("="*60)
        logger.info(f"Total frames: {self.stats['total_frames']}")
        logger.info(f"Removed duplicates: {self.stats['removed_duplicates']}")
        logger.info(f"Unique frames: {self.stats['total_frames'] - self.stats['removed_duplicates']}")
        logger.info(f"Clusters: {self.stats['clusters']}")
        logger.info(f"Selected frames: {self.stats['selected_frames']}")
        logger.info(f"\nSelection rate: {self.stats['selected_frames']/self.stats['total_frames']*100:.1f}%")
        logger.info(f"\n✓ Statistics saved to {stats_file}")

    def run(self):
        """Run frame selection pipeline"""
        logger.info("="*60)
        logger.info("Intelligent Frame Selection for BlindSpot")
        logger.info("="*60 + "\n")

        # Step 1: Find all images
        logger.info(f"Searching for frames in {self.input_dir}...")
        image_paths = self.find_images()

        if not image_paths:
            logger.error(f"No images found in {self.input_dir}")
            return

        self.stats['total_frames'] = len(image_paths)
        logger.info(f"✓ Found {len(image_paths)} frames\n")

        # Step 2: Remove duplicates
        unique_paths = self.remove_duplicates(image_paths)

        if len(unique_paths) <= self.num_frames:
            logger.info(f"\n⚠️  Only {len(unique_paths)} unique frames found")
            logger.info(f"   Requested: {self.num_frames}")
            logger.info(f"   Selecting all unique frames instead")
            self.save_selected_frames(unique_paths)
            self.stats['selected_frames'] = len(unique_paths)
            self.generate_statistics()
            return

        # Step 3: Cluster frames
        labels, kmeans, valid_paths, features = self.cluster_frames(unique_paths)

        # Step 4: Select representative frames
        selected_frames = self.select_representative_frames(
            valid_paths,
            labels,
            features,
            kmeans
        )

        # Step 5: Save selected frames
        self.save_selected_frames(selected_frames)

        # Step 6: Generate statistics
        self.generate_statistics()

        logger.info("\n" + "="*60)
        logger.info("✓ Frame Selection Complete!")
        logger.info("="*60)
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review selected frames in: {self.output_dir}")
        logger.info(f"2. Annotate frames using LabelImg or Roboflow")
        logger.info(f"3. Convert annotations to YOLO format")


def main():
    parser = argparse.ArgumentParser(
        description='Select diverse frames for annotation using intelligent clustering'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing extracted frames'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/custom/selected',
        help='Output directory for selected frames'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=300,
        help='Number of frames to select (default: 300)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=int,
        default=5,
        help='Perceptual hash distance for duplicate detection (default: 5)'
    )

    args = parser.parse_args()

    selector = FrameSelector(
        input_dir=args.input,
        output_dir=args.output,
        num_frames=args.num_frames,
        similarity_threshold=args.similarity_threshold
    )
    selector.run()


if __name__ == '__main__':
    main()
