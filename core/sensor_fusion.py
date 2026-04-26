"""
Sensor fusion helpers for Raspberry Pi hardware.
Combines camera detections with forward ultrasonic and IMU readings.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class SensorFusion:
    """Combines Pi camera detections with ultrasonic and IMU context."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.forward_alert_distance_m = float(self.config.get("forward_alert_distance_m", 0.8))
        self.forward_caution_distance_m = float(self.config.get("forward_caution_distance_m", 1.5))
        self.tilt_warning_deg = float(self.config.get("tilt_warning_deg", 18.0))
        self.last_packet: Dict[str, Any] = {
            "ultrasonic": None,
            "imu": None,
            "timestamp": 0.0,
        }

    def update(self, ultrasonic: Optional[Dict[str, Any]], imu: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self.last_packet = {
            "ultrasonic": ultrasonic,
            "imu": imu,
            "timestamp": time.time(),
        }
        return self.last_packet

    def augment_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach sensor context to detections without changing the detection schema."""
        packet = self.last_packet
        ultrasonic = packet.get("ultrasonic") or {}
        imu = packet.get("imu") or {}
        augmented = []
        for det in detections:
            enriched = det.copy()
            if ultrasonic.get("distance_m") is not None:
                enriched["forward_distance_m"] = ultrasonic["distance_m"]
            if imu:
                enriched["imu_pitch_deg"] = imu.get("pitch_deg")
                enriched["imu_roll_deg"] = imu.get("roll_deg")
            augmented.append(enriched)
        return augmented

    def forward_obstacle_distance(self) -> Optional[float]:
        ultrasonic = self.last_packet.get("ultrasonic") or {}
        return ultrasonic.get("distance_m")

    def forward_blocked(self) -> bool:
        distance = self.forward_obstacle_distance()
        return distance is not None and distance <= self.forward_alert_distance_m

    def forward_caution(self) -> bool:
        distance = self.forward_obstacle_distance()
        return distance is not None and distance <= self.forward_caution_distance_m

    def tilt_warning(self) -> Optional[str]:
        imu = self.last_packet.get("imu") or {}
        pitch = imu.get("pitch_deg")
        roll = imu.get("roll_deg")
        if pitch is None or roll is None:
            return None
        if abs(pitch) >= self.tilt_warning_deg:
            return "pitch"
        if abs(roll) >= self.tilt_warning_deg:
            return "roll"
        return None
