"""
Distance and Direction Calculator Module
Calculates object distance and direction using camera-based estimation.
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional


class DistanceDirectionCalculator:
    """Calculates distance and direction for detected objects."""
    
    def __init__(self, camera_config: dict, distance_config: dict, object_data: dict):
        """
        Initialize calculator.
        
        Args:
            camera_config: Camera configuration (width, height, FOV)
            distance_config: Distance estimation configuration
            object_data: Object metadata with real-world sizes
        """
        self.camera_config = camera_config
        self.distance_config = distance_config
        self.object_sizes = object_data.get('object_sizes', {})
        
        # Camera properties
        self.frame_width = camera_config.get('width', 640)
        self.frame_height = camera_config.get('height', 480)
        self.fov_horizontal = camera_config.get('fov_horizontal', 60)
        
        # Calculate focal length (can be calibrated)
        self.focal_length = distance_config.get('focal_length', 615.0)
        
        # FOV calculations
        self.fov_vertical = self.fov_horizontal * (self.frame_height / self.frame_width)
        
    def calculate_distance(self, detection: Dict) -> float:
        """
        Estimate distance to object using known object sizes.
        
        Args:
            detection: Detection dictionary with class_name and bbox
            
        Returns:
            Estimated distance in meters
        """
        class_name = detection.get('class_name', '')
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Get object metadata
        if class_name not in self.object_sizes:
            # Use average estimate for unknown objects
            return self._fallback_distance_estimate(bbox)
        
        obj_data = self.object_sizes[class_name]
        real_size = obj_data.get('average_height', 1.0)
        use_dimension = obj_data.get('use_dimension', 'height')
        
        # Calculate pixel size
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        pixel_width = x2 - x1
        
        if use_dimension == 'width':
            real_size = obj_data.get('average_width', real_size)
            pixel_size = pixel_width
        else:
            pixel_size = pixel_height
        
        # Avoid division by zero
        if pixel_size < 1:
            return self._fallback_distance_estimate(bbox)
        
        # Distance formula: Distance = (Real_Size × Focal_Length) / Pixel_Size
        distance = (real_size * self.focal_length) / pixel_size
        
        # Clamp to reasonable range (0.1m to 20m)
        distance = max(0.1, min(20.0, distance))
        
        return distance
    
    def _fallback_distance_estimate(self, bbox: list) -> float:
        """
        Fallback distance estimation based on bbox size.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Rough distance estimate in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_width * self.frame_height
        
        # Rough heuristic: larger objects are closer
        area_ratio = bbox_area / frame_area
        
        if area_ratio > 0.3:
            return 1.0  # Very close
        elif area_ratio > 0.1:
            return 2.0
        elif area_ratio > 0.05:
            return 3.5
        else:
            return 5.0  # Far away
    
    def calculate_direction(self, detection: Dict) -> Dict:
        """
        Calculate direction to object based on camera FOV.
        
        Args:
            detection: Detection dictionary with center coordinates
            
        Returns:
            Dictionary with angle_degrees, direction_text, zone
        """
        center = detection.get('center', [self.frame_width // 2, self.frame_height // 2])
        cx, cy = center
        
        # Calculate horizontal angle from center
        frame_center_x = self.frame_width / 2
        pixel_offset = cx - frame_center_x
        
        # Convert pixel offset to angle
        # angle = (pixel_offset / frame_width) * FOV
        angle_degrees = (pixel_offset / self.frame_width) * self.fov_horizontal
        
        # Determine zone (center, left, right)
        zone = self._get_zone(cx)
        direction_text = self._zone_to_direction_text(zone)
        
        # SPECIAL CASE: Large close objects should only be "right in front"
        # when they are also near the image center.
        bbox = detection.get('bbox', [0,0,0,0])
        box_width = bbox[2] - bbox[0]
        width_ratio = box_width / self.frame_width
        
        if width_ratio > 0.4 and abs(angle_degrees) < 8 and zone == 'center':
            direction_text = "center"
        else:
            direction_text = self._zone_to_direction_text(zone)
        
        return {
            'angle_degrees': round(angle_degrees, 1),
            'direction_text': direction_text,
            'zone': zone,
            'position': 'center' if abs(angle_degrees) < 10 else ('left' if angle_degrees < 0 else 'right')
        }
    
    def _zone_to_direction_text(self, zone: str) -> str:
        """Convert zone classification to a clear direction label."""
        if zone == 'left':
            return "left side"
        if zone == 'right':
            return "right side"
        return "center"
    
    def _get_zone(self, x: int) -> str:
        """
        Get zone classification for horizontal position.
        
        Args:
            x: X coordinate in frame
            
        Returns:
            Zone name: 'center', 'left', 'right'
        """
        center_threshold = 0.22  # Narrower center band for clearer direction labels
        left_boundary = self.frame_width * (0.5 - center_threshold / 2)
        right_boundary = self.frame_width * (0.5 + center_threshold / 2)
        
        if left_boundary <= x <= right_boundary:
            return 'center'
        elif x < left_boundary:
            return 'left'
        else:
            return 'right'
    
    def is_in_center_zone(self, detection: Dict, zone_width_percent: float = 0.3) -> bool:
        """
        Check if object is in center zone of view.
        
        Args:
            detection: Detection dictionary
            zone_width_percent: Center zone width as fraction of frame width
            
        Returns:
            True if object center is in center zone
        """
        center = detection.get('center', [self.frame_width // 2, self.frame_height // 2])
        cx = center[0]
        
        left_boundary = self.frame_width * (0.5 - zone_width_percent / 2)
        right_boundary = self.frame_width * (0.5 + zone_width_percent / 2)
        
        return left_boundary <= cx <= right_boundary
    
    def calculate_all(self, detection: Dict) -> Dict:
        """
        Calculate both distance and direction for a detection.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Enhanced detection with distance and direction info
        """
        enhanced = detection.copy()
        
        # Calculate distance
        distance = self.calculate_distance(detection)
        enhanced['distance'] = distance
        
        # Calculate direction
        direction_info = self.calculate_direction(detection)
        enhanced.update(direction_info)
        
        return enhanced


def test_calculator():
    """Test function for distance and direction calculator."""
    import yaml
    import json
    
    # Load configs
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/object_data.json', 'r') as f:
        object_data = json.load(f)
    
    calculator = DistanceDirectionCalculator(
        config['camera'],
        config['distance'],
        object_data
    )
    
    # Test with sample detections
    test_detections = [
        {
            'class_name': 'person',
            'bbox': [200, 100, 350, 400],
            'center': [275, 250],
            'confidence': 0.85
        },
        {
            'class_name': 'chair',
            'bbox': [450, 200, 550, 350],
            'center': [500, 275],
            'confidence': 0.75
        }
    ]
    
    print("Testing Distance & Direction Calculator:")
    print("-" * 50)
    
    for det in test_detections:
        enhanced = calculator.calculate_all(det)
        print(f"\nObject: {enhanced['class_name']}")
        print(f"  Distance: {enhanced['distance']:.2f} meters")
        print(f"  Direction: {enhanced['direction_text']}")
        print(f"  Angle: {enhanced['angle_degrees']}°")
        print(f"  Zone: {enhanced['zone']}")
        print(f"  In center: {calculator.is_in_center_zone(det)}")
    
    print("\nCalculator test completed!")


if __name__ == "__main__":
    test_calculator()
