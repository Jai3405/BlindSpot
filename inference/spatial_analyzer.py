"""
Spatial Analysis Module
Converts depth maps + detections into real-world distance estimates and spatial awareness.
Pure Python/NumPy - no training required!
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class SpatialObject:
    """Represents a detected object with spatial information."""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    depth: float  # Relative depth (0-1, closer = higher)
    distance: float  # Estimated distance in meters
    position: str  # 'left', 'center', 'right'
    priority: int  # Alert priority (1=highest)


class SpatialAnalyzer:
    """Analyzes spatial relationships between detected objects and user."""

    def __init__(self,
                 image_width=640,
                 image_height=480,
                 fov_horizontal=60,  # Field of view in degrees
                 max_distance=10.0,  # Maximum detection distance in meters
                 min_distance=0.5):   # Minimum distance in meters
        """
        Initialize spatial analyzer.

        Args:
            image_width: Input image width
            image_height: Input image height
            fov_horizontal: Horizontal field of view in degrees
            max_distance: Maximum distance to consider (meters)
            min_distance: Minimum distance threshold (meters)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal = fov_horizontal
        self.max_distance = max_distance
        self.min_distance = min_distance

        # Calculate zones
        self.left_zone = (0, image_width // 3)
        self.center_zone = (image_width // 3, 2 * image_width // 3)
        self.right_zone = (2 * image_width // 3, image_width)

        # Priority thresholds (meters) - MORE AGGRESSIVE for safety
        self.critical_distance = 2.5  # < 2.5m = critical (increased for earlier warning)
        self.warning_distance = 4.0   # < 4.0m = warning
        self.info_distance = 6.0      # < 6.0m = info

    def depth_to_distance(self, depth_value, depth_map):
        """
        Convert relative depth value to estimated distance in meters.

        Uses inverse relationship: closer objects have higher depth values.

        Args:
            depth_value: Relative depth (from MiDaS)
            depth_map: Full depth map for normalization

        Returns:
            distance: Estimated distance in meters
        """
        # Normalize depth value
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            normalized_depth = (depth_value - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = 0.5

        # Inverse mapping: high depth = close, low depth = far
        # Using exponential for better near-field resolution
        distance = self.min_distance + (self.max_distance - self.min_distance) * (1 - normalized_depth) ** 2

        return float(np.clip(distance, self.min_distance, self.max_distance))

    def get_object_position(self, bbox):
        """
        Determine if object is on left, center, or right.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            position: 'left', 'center', or 'right'
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2

        if center_x < self.left_zone[1]:
            return 'left'
        elif center_x < self.center_zone[1]:
            return 'center'
        else:
            return 'right'

    def calculate_priority(self, distance, class_name):
        """
        Calculate alert priority based on distance and object type.

        Args:
            distance: Distance in meters
            class_name: Object class name

        Returns:
            priority: 1 (critical), 2 (warning), 3 (info), 4 (low)
        """
        # Critical obstacles (always high priority if close)
        critical_objects = {'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'}

        # Medium priority
        medium_objects = {'chair', 'couch', 'bed', 'dining table', 'tv'}

        if distance < self.critical_distance:
            return 1  # Critical
        elif distance < self.warning_distance:
            if class_name in critical_objects:
                return 1  # Still critical for moving obstacles
            else:
                return 2  # Warning
        elif distance < self.info_distance:
            if class_name in critical_objects:
                return 2  # Warning for far but important objects
            elif class_name in medium_objects:
                return 3  # Info
            else:
                return 4  # Low priority
        else:
            return 4  # Low priority (far away)

    def analyze_detections(self,
                          detections: List[Dict],
                          depth_map: np.ndarray,
                          depth_method: str = 'min') -> List[SpatialObject]:
        """
        Analyze detected objects with depth information.

        Args:
            detections: List of YOLO detections
                Each detection: {'bbox': [x1,y1,x2,y2], 'class': str, 'conf': float, 'class_id': int}
            depth_map: Depth map from MiDaS
            depth_method: Method to extract depth from bbox ('min', 'mean', 'center')

        Returns:
            spatial_objects: List of SpatialObject with distance/position info
        """
        spatial_objects = []

        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['conf']
            class_id = det.get('class_id', 0)

            # Get depth value in bounding box
            depth_value = self._get_bbox_depth(depth_map, bbox, method=depth_method)

            # Convert to distance
            distance = self.depth_to_distance(depth_value, depth_map)

            # Get position
            position = self.get_object_position(bbox)

            # Calculate priority
            priority = self.calculate_priority(distance, class_name)

            # Create spatial object
            spatial_obj = SpatialObject(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                bbox=tuple(bbox),
                depth=depth_value,
                distance=distance,
                position=position,
                priority=priority
            )

            spatial_objects.append(spatial_obj)

        # Sort by priority (highest first), then by distance (closest first)
        spatial_objects.sort(key=lambda x: (x.priority, x.distance))

        return spatial_objects

    def _get_bbox_depth(self, depth_map, bbox, method='min'):
        """Extract depth value from bounding box region."""
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
            return float(region.min())
        elif method == 'mean':
            return float(region.mean())
        elif method == 'center':
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            return float(depth_map[cy, cx])
        else:
            return float(region.min())

    def get_navigation_hints(self, spatial_objects: List[SpatialObject]) -> Dict:
        """
        Generate navigation hints based on spatial objects.

        Args:
            spatial_objects: List of SpatialObject

        Returns:
            hints: Dictionary with navigation recommendations
        """
        hints = {
            'safe_direction': None,
            'obstacles_ahead': [],
            'critical_alerts': [],
            'recommendations': []
        }

        # Count obstacles per zone
        left_obstacles = [obj for obj in spatial_objects if obj.position == 'left' and obj.priority <= 2]
        center_obstacles = [obj for obj in spatial_objects if obj.position == 'center' and obj.priority <= 2]
        right_obstacles = [obj for obj in spatial_objects if obj.position == 'right' and obj.priority <= 2]

        # Find safest direction
        obstacle_counts = {
            'left': len(left_obstacles),
            'center': len(center_obstacles),
            'right': len(right_obstacles)
        }

        hints['safe_direction'] = min(obstacle_counts, key=obstacle_counts.get)

        # Critical alerts (priority 1)
        critical = [obj for obj in spatial_objects if obj.priority == 1]
        hints['critical_alerts'] = [
            f"{obj.class_name} {obj.position} at {obj.distance:.1f}m"
            for obj in critical[:3]  # Top 3 critical
        ]

        # Obstacles directly ahead
        ahead = [obj for obj in center_obstacles if obj.distance < 3.0]
        hints['obstacles_ahead'] = [
            f"{obj.class_name} at {obj.distance:.1f}m"
            for obj in ahead[:2]
        ]

        # Generate recommendations
        if critical:
            hints['recommendations'].append("STOP - Critical obstacle detected")
        elif len(center_obstacles) > 0 and center_obstacles[0].distance < 2.0:
            if obstacle_counts['left'] < obstacle_counts['right']:
                hints['recommendations'].append("Move left to avoid obstacle")
            elif obstacle_counts['right'] < obstacle_counts['left']:
                hints['recommendations'].append("Move right to avoid obstacle")
            else:
                hints['recommendations'].append("Slow down - obstacle ahead")
        elif len(spatial_objects) == 0:
            hints['recommendations'].append("Path is clear")

        return hints


def test_spatial_analyzer():
    """Test spatial analyzer with mock data."""
    print("Testing SpatialAnalyzer...")

    # Create analyzer
    analyzer = SpatialAnalyzer(image_width=640, image_height=480)

    # Mock depth map
    depth_map = np.random.rand(480, 640) * 100

    # Mock detections
    detections = [
        {'bbox': [100, 100, 200, 300], 'class': 'person', 'conf': 0.9, 'class_id': 0},
        {'bbox': [300, 150, 400, 350], 'class': 'chair', 'conf': 0.8, 'class_id': 56},
        {'bbox': [500, 200, 600, 400], 'class': 'car', 'conf': 0.95, 'class_id': 2},
    ]

    # Analyze
    spatial_objects = analyzer.analyze_detections(detections, depth_map)

    print(f"\nDetected {len(spatial_objects)} objects:")
    for obj in spatial_objects:
        print(f"  [{obj.priority}] {obj.class_name} - {obj.position} - {obj.distance:.2f}m")

    # Get navigation hints
    hints = analyzer.get_navigation_hints(spatial_objects)

    print(f"\nNavigation hints:")
    print(f"  Safe direction: {hints['safe_direction']}")
    print(f"  Critical alerts: {hints['critical_alerts']}")
    print(f"  Recommendations: {hints['recommendations']}")

    print("\n✓ SpatialAnalyzer test passed!")


if __name__ == "__main__":
    test_spatial_analyzer()
