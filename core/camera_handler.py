"""
Camera Handler Module
Manages camera initialization, frame capture, and camera properties.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple


class CameraHandler:
    """Handles camera capture and frame management."""
    
    def __init__(self, config: dict):
        """
        Initialize camera handler.
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config
        self.camera = None
        self.current_frame = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        
        # Camera properties
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps_target = config.get('fps_target', 24)
        self.fov_horizontal = config.get('fov_horizontal', 60)
        self.device_id = config.get('device_id', 0)
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
    def initialize(self) -> bool:
        """
        Initialize the camera.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"[CameraHandler] Initializing camera {self.device_id}...")
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                print("[CameraHandler] ERROR: Could not open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[CameraHandler] Camera initialized: {actual_width}x{actual_height}")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[CameraHandler] ERROR: Failed to initialize camera: {e}")
            return False
    
    def _capture_loop(self):
        """Internal loop for continuous frame capture."""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
                    self.frame_count += 1
                    
                # Update FPS
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
            else:
                print("[CameraHandler] WARNING: Failed to read frame")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest camera frame.
        
        Returns:
            numpy.ndarray: Latest frame or None if no frame available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            float: Current frames per second
        """
        return self.fps
    
    def get_properties(self) -> dict:
        """
        Get camera properties.
        
        Returns:
            dict: Camera properties including FOV, resolution, etc.
        """
        return {
            'width': self.width,
            'height': self.height,
            'fov_horizontal': self.fov_horizontal,
            'fov_vertical': self.fov_horizontal * (self.height / self.width),
            'fps_target': self.fps_target,
            'fps_actual': self.fps,
            'device_id': self.device_id
        }
    
    def release(self):
        """Release camera resources."""
        print("[CameraHandler] Releasing camera...")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        print("[CameraHandler] Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()


def test_camera():
    """Test function for camera handler."""
    import yaml
    
    # Load config
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera = CameraHandler(config['camera'])
    
    if camera.initialize():
        print("Camera test: Displaying feed for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            frame = camera.get_frame()
            if frame is not None:
                # Add FPS text
                fps_text = f"FPS: {camera.get_fps():.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Camera Test', frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        camera.release()
        print("Camera test completed!")
    else:
        print("Camera test failed!")


if __name__ == "__main__":
    test_camera()
