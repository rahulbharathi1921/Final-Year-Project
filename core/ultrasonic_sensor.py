"""
Ultrasonic sensor support for Raspberry Pi.
Reads forward obstacle distance from an HC-SR04-style sensor.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

try:
    from gpiozero import DistanceSensor
except Exception:  # pragma: no cover - unavailable on laptop/dev systems
    DistanceSensor = None


class UltrasonicSensor:
    """Forward distance sensor wrapper with smoothing and graceful fallback."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.trigger_pin = int(self.config.get("trigger_pin", 23))
        self.echo_pin = int(self.config.get("echo_pin", 24))
        self.max_distance_m = float(self.config.get("max_distance_m", 3.0))
        self.sample_count = int(self.config.get("sample_count", 3))
        self.offset_m = float(self.config.get("offset_m", 0.0))
        self.sensor = None
        self.available = False
        self.last_distance_m: Optional[float] = None
        self.last_read_time = 0.0

    def initialize(self) -> bool:
        """Initialize GPIO-backed sensor if available."""
        if not self.enabled:
            return False
        if DistanceSensor is None:
            print("[Ultrasonic] gpiozero not available; sensor disabled.")
            return False

        try:
            self.sensor = DistanceSensor(
                echo=self.echo_pin,
                trigger=self.trigger_pin,
                max_distance=self.max_distance_m,
            )
            self.available = True
            print(f"[Ultrasonic] Initialized on trigger={self.trigger_pin}, echo={self.echo_pin}")
            return True
        except Exception as exc:
            print(f"[Ultrasonic] Initialization failed: {exc}")
            self.sensor = None
            self.available = False
            return False

    def read(self) -> Optional[Dict[str, Any]]:
        """Read filtered distance in meters."""
        if not self.available or self.sensor is None:
            return None

        samples: List[float] = []
        for _ in range(max(1, self.sample_count)):
            try:
                distance = float(self.sensor.distance) * self.max_distance_m
            except Exception:
                distance = 0.0
            if 0.02 <= distance <= self.max_distance_m:
                samples.append(distance)
            time.sleep(0.01)

        if not samples:
            return None

        filtered = statistics.median(samples) + self.offset_m
        filtered = max(0.02, min(self.max_distance_m, filtered))
        self.last_distance_m = filtered
        self.last_read_time = time.time()
        return {
            "distance_m": filtered,
            "samples": len(samples),
            "timestamp": self.last_read_time,
        }

    def shutdown(self) -> None:
        """Release GPIO resources."""
        if self.sensor is not None:
            try:
                self.sensor.close()
            except Exception:
                pass
        self.sensor = None
        self.available = False
