"""
Simple Object Tracker Module
Basic frame-to-frame tracking using centroid matching.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class SimpleTracker:
    """
    Simple object tracker using centroid matching.
    Tracks objects across frames using IoU and centroid distance.
    """
    
    def __init__(self, max_disappeared: int = 5, max_distance: float = 120.0):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Max frames an object can disappear before removal
            max_distance: Max pixel distance for centroid matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track state
        self.next_object_id = 0
        self.objects = {}  # {object_id: {detection_data}}
        self.disappeared = {}  # {object_id: frame_count}

    def _build_track_state(self, detection: Dict[str, Any], track_id: int, previous_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Apply class-vote smoothing so one bad frame does not rename the object."""
        state = detection.copy()
        votes = dict((previous_state or {}).get('_class_votes', {}))
        raw_class_name = detection.get('raw_class_name') or detection.get('class_name') or detection.get('class')
        previous_class_name = (previous_state or {}).get('class_name')

        if raw_class_name:
            votes[raw_class_name] = votes.get(raw_class_name, 0) + 1

        if votes:
            if previous_class_name and votes.get(previous_class_name, 0) == max(votes.values()):
                stable_class_name = previous_class_name
            else:
                stable_class_name = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        else:
            stable_class_name = raw_class_name

        stable_class_name = stable_class_name or raw_class_name or detection.get('class') or 'object'
        state['track_id'] = track_id
        state['object_id'] = track_id
        state['class'] = stable_class_name
        state['raw_class_name'] = raw_class_name
        state['class_name'] = stable_class_name
        state['_class_votes'] = votes
        return state
        
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of current frame detections
            
        Returns:
            Detections with added 'track_id' field
        """
        # If no current detections
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            
            return []
        
        # If no existing tracked objects, register all detections as new
        if len(self.objects) == 0:
            tracked_detections = []
            for det in detections:
                track_id = self.next_object_id
                track_state = self._build_track_state(det, track_id)
                self.objects[track_id] = track_state
                self.next_object_id += 1
                tracked_detections.append(track_state.copy())
            
            return tracked_detections
        
        # Match detections to existing tracked objects
        object_ids = list(self.objects.keys())
        object_centroids = np.array([
            self._get_centroid(self.objects[obj_id]['bbox']) 
            for obj_id in object_ids
        ])
        
        detection_centroids = np.array([
            self._get_centroid(det['bbox']) 
            for det in detections
        ])
        
        # Calculate global assignment cost matrix.
        distances = cdist(object_centroids, detection_centroids)
        costs = np.full(distances.shape, np.inf, dtype=float)
        for obj_idx, object_id in enumerate(object_ids):
            previous_state = self.objects[object_id]
            previous_class = previous_state.get('class_name') or previous_state.get('class')
            previous_bbox = previous_state.get('bbox')
            for det_idx, det in enumerate(detections):
                distance = distances[obj_idx, det_idx]
                if distance > self.max_distance:
                    continue

                current_class = det.get('class_name') or det.get('class')
                class_penalty = 35.0 if previous_class and current_class and previous_class != current_class else 0.0
                iou = self._calculate_iou(previous_bbox, det['bbox']) if previous_bbox else 0.0
                costs[obj_idx, det_idx] = distance + class_penalty - (iou * 30.0)
        
        # Find best matches
        matched_objects = set()
        matched_detections = set()
        tracked_detections = []

        if costs.size > 0 and np.isfinite(costs).any():
            row_indices, col_indices = linear_sum_assignment(np.where(np.isfinite(costs), costs, 1e9))
            for obj_idx, det_idx in zip(row_indices, col_indices):
                if not np.isfinite(costs[obj_idx, det_idx]):
                    continue

                object_id = object_ids[obj_idx]
                matched_objects.add(object_id)
                matched_detections.add(det_idx)

                previous_state = self.objects.get(object_id)
                track_state = self._build_track_state(detections[det_idx], object_id, previous_state)
                self.objects[object_id] = track_state
                if object_id in self.disappeared:
                    del self.disappeared[object_id]

                tracked_detections.append(track_state.copy())
        
        # Handle unmatched objects (disappeared)
        for i, object_id in enumerate(object_ids):
            if object_id not in matched_objects:
                if object_id not in self.disappeared:
                    self.disappeared[object_id] = 1
                else:
                    self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    if object_id in self.disappeared:
                        del self.disappeared[object_id]
        
        # Register new objects
        for i, det in enumerate(detections):
            if i not in matched_detections:
                track_id = self.next_object_id
                track_state = self._build_track_state(det, track_id)
                self.objects[track_id] = track_state
                self.next_object_id += 1
                tracked_detections.append(track_state.copy())
        
        return tracked_detections
    
    def get_tracked_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self.objects)
    
    def reset(self):
        """Reset tracker state."""
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0


def test_tracker():
    """Test function for tracker."""
    print("=== Simple Tracker Test ===\n")
    
    tracker = SimpleTracker()
    
    # Simulate detection frames
    frames = [
        # Frame 1: Two objects
        [
            {'class': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.9},
            {'class': 'laptop', 'bbox': [300, 150, 450, 250], 'confidence': 0.85}
        ],
        # Frame 2: Same objects, slightly moved
        [
            {'class': 'person', 'bbox': [105, 105, 205, 305], 'confidence': 0.9},
            {'class': 'laptop', 'bbox': [305, 155, 455, 255], 'confidence': 0.85}
        ],
        # Frame 3: Only person (laptop disappeared)
        [
            {'class': 'person', 'bbox': [110, 110, 210, 310], 'confidence': 0.9}
        ],
        # Frame 4: Person + new chair
        [
            {'class': 'person', 'bbox': [115, 115, 215, 315], 'confidence': 0.9},
            {'class': 'chair', 'bbox': [500, 200, 600, 400], 'confidence': 0.8}
        ]
    ]
    
    for i, frame_detections in enumerate(frames):
        print(f"Frame {i + 1}:")
        tracked = tracker.update(frame_detections)
        
        for det in tracked:
            track_id = det.get('track_id', -1)
            print(f"  Track ID {track_id}: {det['class']} at {det['bbox']}")
        
        print(f"  Total tracked: {tracker.get_tracked_count()}\n")
    
    print("✅ Tracker test completed!")


if __name__ == "__main__":
    test_tracker()
