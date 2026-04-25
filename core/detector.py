"""
Object Detector Module
YOLO wrapper for real-time object detection.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class ObjectDetector:
    """YOLO object detector wrapper."""
    
    def __init__(self, config: dict, object_data: dict):
        """
        Initialize object detector.
        
        Args:
            config: Detection configuration dictionary
            object_data: Object metadata (sizes, classes)
        """
        self.config = config
        self.object_data = object_data
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Detection parameters
        # Detection parameters
        self.model_path = config.get('model_path', 'models/yolov5s.pt')
        self.confidence = config.get('confidence_threshold', config.get('confidence', 0.45))
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_detections = config.get('max_detections', 20)
        
        self.coco_classes = object_data.get('coco_classes', [])
        self.class_names = {}
        self.backend = None
        
    def initialize(self) -> bool:
        """
        Load YOLO model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"[Detector] Loading YOLO model from {self.model_path}...")
            print(f"[Detector] Using device: {self.device}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                print(f"[Detector] ERROR: Model file not found at {self.model_path}")
                print("[Detector] Please move your YOLO model file to the models/ directory")
                return False
            
            # Prefer the model's native class map so labels stay correct.
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.backend = 'ultralytics'
            except Exception:
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=self.model_path,
                    force_reload=False
                )
                self.backend = 'yolov5'
                
            # Configure model
            if self.backend == 'ultralytics':
                if hasattr(self.model, 'overrides'):
                    self.model.overrides['conf'] = self.confidence
                    self.model.overrides['iou'] = self.iou_threshold
                    self.model.overrides['max_det'] = self.max_detections
                names_attr = getattr(self.model, 'names', {})
                if isinstance(names_attr, list):
                    self.class_names = {i: name for i, name in enumerate(names_attr)}
                else:
                    self.class_names = dict(names_attr or {})
            else:
                self.model.conf = self.confidence
                self.model.iou = self.iou_threshold
                self.model.max_det = self.max_detections
                names_attr = getattr(self.model, 'names', {})
                if isinstance(names_attr, list):
                    self.class_names = {i: name for i, name in enumerate(names_attr)}
                else:
                    self.class_names = dict(names_attr or {})
            
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            print("[Detector] Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"[Detector] ERROR: Failed to load model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image frame (BGR)
            
        Returns:
            List of detection dictionaries with keys:
                - class_name: Object class name
                - confidence: Detection confidence (0-1)
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - center: [cx, cy] center point
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            if self.backend == 'ultralytics':
                results = self.model(
                    frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )
            else:
                results = self.model(frame)
            
            detections = []
            
            # Parse results (handling both YOLOv5 and Ultralytics YOLO formats)
            if self.backend == 'yolov5' and hasattr(results, 'pandas'):
                # YOLOv5 format
                try:
                    df = results.pandas().xyxy[0]
                except (IndexError, AttributeError, KeyError):
                    return []
                if df is None or df.empty:
                    return []
                for _, row in df.iterrows():
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    detection = {
                        'class_name': row['name'],
                        'raw_class_name': row['name'],
                        'confidence': float(row['confidence']),
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'class_id': int(row['class'])
                    }
                    detections.append(detection)
                    
            else:
                # Ultralytics YOLO format
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            'class_name': class_name,
                            'raw_class_name': class_name,
                            'confidence': float(box.conf[0]),
                            'bbox': [x1, y1, x2, y2],
                            'center': [cx, cy],
                            'class_id': class_id
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[Detector] ERROR during detection: {e}")
            return []
    
    def filter_by_mode(self, detections: List[Dict], mode: str, modes_config: dict) -> List[Dict]:
        """
        Filter detections based on current mode.
        
        Args:
            detections: List of detection dictionaries
            mode: Current mode (indoor/outdoor/jarvis)
            modes_config: Mode configuration from settings
            
        Returns:
            Filtered list of detections
        """
        if mode not in modes_config:
            return detections
        
        mode_cfg = modes_config[mode]
        
        # Jarvis mode shows all detections
        if mode_cfg.get('full_detection', False):
            return detections
        
        # Filter by focus objects
        focus_objects = mode_cfg.get('focus_objects') or []
        if not focus_objects:
            return detections
        focus_set = set(focus_objects)
        filtered = [d for d in detections if d.get('class_name', '') in focus_set]
        return filtered
    
    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported object classes.
        
        Returns:
            List of class names
        """
        return self.coco_classes.copy()


def test_detector():
    """Test function for object detector."""
    import yaml
    import json
    
    # Load configs
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/object_data.json', 'r') as f:
        object_data = json.load(f)
    
    detector = ObjectDetector(config['detection'], object_data)
    
    if detector.initialize():
        print("Detector test: Using sample image...")
        
        # Create a test image or use webcam
        import cv2
        cap = cv2.VideoCapture(0)
        
        ret, frame = cap.read()
        if ret:
            detections = detector.detect(frame)
            print(f"\nDetected {len(detections)} objects:")
            
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Detection Test', frame)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        
        cap.release()
        print("Detector test completed!")
    else:
        print("Detector test failed!")


if __name__ == "__main__":
    test_detector()
