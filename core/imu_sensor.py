"""
MPU6050 IMU support for Raspberry Pi.
Provides basic accelerometer/gyro reads and tilt estimation.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

try:
    from smbus2 import SMBus
except Exception:  # pragma: no cover - unavailable on laptop/dev systems
    SMBus = None


class MPU6050Sensor:
    """Minimal MPU6050 reader with tilt estimation."""

    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.bus_id = int(self.config.get("bus", 1))
        self.address = int(str(self.config.get("address", "0x68")), 0)
        self.available = False
        self.bus = None
        self.last_reading: Optional[Dict[str, Any]] = None

    def initialize(self) -> bool:
        if not self.enabled:
            return False
        if SMBus is None:
            print("[IMU] smbus2 not available; IMU disabled.")
            return False

        try:
            self.bus = SMBus(self.bus_id)
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
            self.available = True
            print(f"[IMU] MPU6050 initialized on I2C bus {self.bus_id} address {hex(self.address)}")
            return True
        except Exception as exc:
            print(f"[IMU] Initialization failed: {exc}")
            self.bus = None
            self.available = False
            return False

    def _read_word_signed(self, reg: int) -> int:
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        return value - 65536 if value > 32767 else value

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.available or self.bus is None:
            return None

        try:
            accel_x = self._read_word_signed(self.ACCEL_XOUT_H) / 16384.0
            accel_y = self._read_word_signed(self.ACCEL_XOUT_H + 2) / 16384.0
            accel_z = self._read_word_signed(self.ACCEL_XOUT_H + 4) / 16384.0
            gyro_x = self._read_word_signed(self.GYRO_XOUT_H) / 131.0
            gyro_y = self._read_word_signed(self.GYRO_XOUT_H + 2) / 131.0
            gyro_z = self._read_word_signed(self.GYRO_XOUT_H + 4) / 131.0

            pitch = math.degrees(math.atan2(accel_x, math.sqrt(accel_y ** 2 + accel_z ** 2)))
            roll = math.degrees(math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)))

            self.last_reading = {
                "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
                "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
                "pitch_deg": pitch,
                "roll_deg": roll,
                "timestamp": time.time(),
            }
            return self.last_reading
        except Exception as exc:
            print(f"[IMU] Read failed: {exc}")
            return None

    def shutdown(self) -> None:
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
        self.bus = None
        self.available = False
