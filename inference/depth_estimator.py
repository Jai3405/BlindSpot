"""
MiDaS Depth Estimation Module
Uses pre-trained MiDaS model for monocular depth estimation.
No training required - works out of the box!
"""

import torch
import cv2
import numpy as np
from pathlib import Path


class DepthEstimator:
    """Monocular depth estimation using MiDaS."""

    def __init__(self, model_type='DPT_Large', device=None):
        """
        Initialize MiDaS depth estimator.

        Args:
            model_type: MiDaS model variant
                - 'DPT_Large': Best quality (slower)
                - 'DPT_Hybrid': Balanced
                - 'MiDaS_small': Fastest (lower quality)
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        self.model_type = model_type

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        print(f"DepthEstimator using device: {self.device}")

        # Load pre-trained MiDaS model
        print(f"Loading MiDaS model: {model_type}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print("✓ MiDaS model loaded successfully!")

    def estimate_depth(self, image):
        """
        Estimate depth map from RGB image.

        Args:
            image: RGB image (H, W, 3) as numpy array or PIL Image

        Returns:
            depth_map: Depth map (H, W) - higher values = closer
        """
        # Convert to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Prepare input
        input_batch = self.transform(image).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        return depth_map

    def get_normalized_depth(self, depth_map, min_val=0, max_val=255):
        """
        Normalize depth map to specified range.

        Args:
            depth_map: Raw depth map
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization

        Returns:
            normalized_depth: Normalized depth map
        """
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
            normalized = normalized * (max_val - min_val) + min_val
        else:
            normalized = np.zeros_like(depth_map)

        return normalized.astype(np.uint8)

    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_INFERNO):
        """
        Create colorized visualization of depth map.

        Args:
            depth_map: Depth map to visualize
            colormap: OpenCV colormap to use

        Returns:
            colored_depth: Colorized depth map (H, W, 3)
        """
        # Normalize to 0-255
        normalized = self.get_normalized_depth(depth_map)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)

        return colored

    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at specific pixel coordinates.

        Args:
            depth_map: Depth map
            x, y: Pixel coordinates

        Returns:
            depth_value: Depth at (x, y)
        """
        h, w = depth_map.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        return depth_map[y, x]

    def get_depth_in_bbox(self, depth_map, bbox, method='min'):
        """
        Get depth value within bounding box.

        Args:
            depth_map: Depth map
            bbox: Bounding box [x1, y1, x2, y2]
            method: 'min', 'mean', 'median', or 'center'
                - 'min': Closest point in bbox
                - 'mean': Average depth
                - 'median': Median depth
                - 'center': Depth at center of bbox

        Returns:
            depth_value: Depth within bbox
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clip to image bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Extract region
        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        # Calculate depth based on method
        if method == 'min':
            # Minimum depth = closest point
            return float(region.min())
        elif method == 'mean':
            return float(region.mean())
        elif method == 'median':
            return float(np.median(region))
        elif method == 'center':
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            return self.get_depth_at_point(depth_map, cx, cy)
        else:
            raise ValueError(f"Unknown method: {method}")


def test_depth_estimator():
    """Test depth estimator with sample image."""
    import matplotlib.pyplot as plt

    # Create sample image or load from file
    print("Testing DepthEstimator...")

    # Initialize estimator
    estimator = DepthEstimator(model_type='MiDaS_small')  # Use small for testing

    # Create test image (gradient)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Estimate depth
    depth_map = estimator.estimate_depth(test_image)

    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")

    # Visualize
    colored_depth = estimator.visualize_depth(depth_map)

    # Test bbox depth
    bbox = [100, 100, 300, 300]
    depth_min = estimator.get_depth_in_bbox(depth_map, bbox, method='min')
    print(f"Depth in bbox (min): {depth_min:.2f}")

    print("✓ DepthEstimator test passed!")

    return estimator, depth_map, colored_depth


if __name__ == "__main__":
    test_depth_estimator()
