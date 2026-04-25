"""
Alert System Module
Manages danger detection, priority scoring, and alert cooldown.
"""

import time
from typing import List, Dict, Optional
from collections import defaultdict


class AlertSystem:
    """Manages danger alerts with priority and cooldown."""
    
    def __init__(self, alert_config: dict, object_data: dict):
        """
        Initialize alert system.
        
        Args:
            alert_config: Alert configuration from settings
            object_data: Object metadata with danger categories
        """
        self.config = alert_config
        self.object_data = object_data
        
        # Alert configuration
        self.cooldown_seconds = alert_config.get('cooldown_seconds', 2.0)
        self.high_priority_distance = alert_config.get('high_priority_distance', 1.0)
        self.medium_priority_distance = alert_config.get('medium_priority_distance', 3.0)
        self.danger_thresholds = alert_config.get('danger_thresholds', {})
        
        # Danger categories
        self.danger_categories = object_data.get('danger_categories', {})
        
        # Cooldown tracking
        self.last_alert_time = defaultdict(float)  # Key: object_type, Value: timestamp
        
        # Alert history
        self.recent_alerts = []
    
    def evaluate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Evaluate detections and determine which need alerts.
        
        Args:
            detections: List of detections with distance and direction info
            
        Returns:
            List of alert dictionaries to announce
        """
        alerts = []
        current_time = time.time()
        
        for det in detections:
            alert = self._evaluate_detection(det, current_time)
            if alert:
                alerts.append(alert)
        
        # Sort by priority (highest first)
        alerts.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return alerts
    
    def _evaluate_detection(self, detection: Dict, current_time: float) -> Optional[Dict]:
        """
        Evaluate single detection for alert.
        
        Args:
            detection: Detection with distance and direction
            current_time: Current timestamp
            
        Returns:
            Alert dictionary or None
        """
        class_name = detection.get('class_name', '')
        distance = detection.get('distance', 999)
        zone = detection.get('zone', 'unknown')
        direction_text = detection.get('direction_text', 'unknown')
        
        # Determine priority level
        priority = self._get_priority_level(class_name, distance, zone)
        
        if priority == 'none':
            return None
        
        # Check cooldown
        alert_key = f"{class_name}_{zone}"
        time_since_last = current_time - self.last_alert_time.get(alert_key, 0)
        
        if time_since_last < self.cooldown_seconds:
            return None  # Still in cooldown
        
        # Update cooldown
        self.last_alert_time[alert_key] = current_time
        
        # Create alert
        alert = {
            'class_name': class_name,
            'distance': distance,
            'direction_text': direction_text,
            'zone': zone,
            'priority': priority,
            'priority_score': self._calculate_priority_score(priority, distance),
            'message': self._generate_alert_message(class_name, distance, direction_text, priority),
            'timestamp': current_time
        }
        
        self.recent_alerts.append(alert)
        
        # Keep only recent 50 alerts
        if len(self.recent_alerts) > 50:
            self.recent_alerts = self.recent_alerts[-50:]
        
        return alert
    
    def _get_priority_level(self, class_name: str, distance: float, zone: str) -> str:
        """
        Determine priority level for object.
        
        Args:
            class_name: Object class name
            distance: Distance in meters
            zone: Zone (center/left/right)
            
        Returns:
            Priority level: 'high', 'medium', 'low', or 'none'
        """
        # Check danger thresholds
        threshold = self.danger_thresholds.get(class_name, None)
        
        # Immediate danger objects
        if class_name in self.danger_categories.get('immediate', []):
            if distance < 2.0 and zone == 'center':
                return 'high'
            elif distance < 3.0:
                return 'medium'
            else:
                return 'low'
        
        # High danger objects
        if class_name in self.danger_categories.get('high', []):
            if threshold and distance < threshold and zone == 'center':
                return 'high'
            elif distance < self.high_priority_distance:
                return 'medium'
            elif distance < self.medium_priority_distance:
                return 'low'
        
        # Medium danger objects
        if class_name in self.danger_categories.get('medium', []):
            if distance < 1.0 and zone == 'center':
                return 'medium'
            elif distance < 2.5:
                return 'low'
        
        # Low danger objects - only report if very close
        if distance < 0.5:
            return 'low'
        
        return 'none'
    
    def _calculate_priority_score(self, priority: str, distance: float) -> float:
        """
        Calculate numerical priority score.
        
        Args:
            priority: Priority level
            distance: Distance in meters
            
        Returns:
            Priority score (higher = more urgent)
        """
        base_scores = {
            'high': 100,
            'medium': 50,
            'low': 20
        }
        
        base = base_scores.get(priority, 0)
        
        # Closer objects get higher scores
        distance_score = max(0, 10 - distance) * 5
        
        return base + distance_score
    
    def _generate_alert_message(self, class_name: str, distance: float, 
                                direction: str, priority: str) -> str:
        """
        Generate natural language alert message.
        
        Args:
            class_name: Object class name
            distance: Distance in meters
            direction: Direction text
            priority: Priority level
            
        Returns:
            Alert message string
        """
        # Priority prefixes
        prefixes = {
            'high': 'Warning:',
            'medium': 'Caution:',
            'low': 'Notice:'
        }
        
        prefix = prefixes.get(priority, '')
        
        # Format distance
        if distance < 1.0:
            dist_text = f"{int(distance * 100)} centimeters"
        else:
            dist_text = f"{distance:.1f} meters"
        
        # Build message
        message = f"{prefix} {class_name} at {dist_text}, {direction}".strip()
        
        return message
    
    def should_speak_alert(self, alert: Dict) -> bool:
        """
        Determine if alert should be spoken immediately.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if should speak now
        """
        # High priority always speaks
        if alert['priority'] == 'high':
            return True
        
        # Medium priority speaks if in center zone
        if alert['priority'] == 'medium' and alert['zone'] == 'center':
            return True
        
        # Low priority doesn't auto-speak (only on query)
        return False
    
    def get_recent_alerts(self, max_count: int = 10) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            max_count: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.recent_alerts[-max_count:]
    
    def clear_cooldown(self, class_name: Optional[str] = None):
        """
        Clear cooldown timers.
        
        Args:
            class_name: Specific class to clear, or None for all
        """
        if class_name:
            keys_to_clear = [k for k in self.last_alert_time.keys() if k.startswith(class_name)]
            for key in keys_to_clear:
                del self.last_alert_time[key]
        else:
            self.last_alert_time.clear()


def test_alert_system():
    """Test function for alert system."""
    import yaml
    import json
    
    # Load configs
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/object_data.json', 'r') as f:
        object_data = json.load(f)
    
    alert_system = AlertSystem(config['alerts'], object_data)
    
    # Test detections
    test_detections = [
        {
            'class_name': 'person',
            'distance': 0.8,
            'direction_text': 'center',
            'zone': 'center',
            'confidence': 0.9
        },
        {
            'class_name': 'car',
            'distance': 2.5,
            'direction_text': '15 degrees left',
            'zone': 'left',
            'confidence': 0.85
        },
        {
            'class_name': 'chair',
            'distance': 1.5,
            'direction_text': 'slightly right',
            'zone': 'right',
            'confidence': 0.75
        }
    ]
    
    print("Testing Alert System:")
    print("-" * 60)
    
    alerts = alert_system.evaluate_detections(test_detections)
    
    print(f"\nGenerated {len(alerts)} alerts:")
    for i, alert in enumerate(alerts, 1):
        print(f"\n{i}. {alert['message']}")
        print(f"   Priority: {alert['priority']} (score: {alert['priority_score']:.1f})")
        print(f"   Should speak: {alert_system.should_speak_alert(alert)}")
    
    print("\nAlert system test completed!")


if __name__ == "__main__":
    test_alert_system()
