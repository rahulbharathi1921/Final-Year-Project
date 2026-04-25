"""
Session logging for NavSense.
Writes human-readable logs and JSONL event streams per run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class SessionLogger:
    """Structured file logger for detections, commands, alerts, and responses."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.log_directory = Path(self.config.get("log_directory", "logs"))
        self.session_prefix = self.config.get("session_prefix", "session_")
        self.save_detections = bool(self.config.get("save_detections", True))
        self.save_alerts = bool(self.config.get("save_alerts", True))
        self.save_voice_commands = bool(self.config.get("save_voice_commands", True))

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_directory / f"{self.session_prefix}{self.session_id}"
        self.logger = logging.getLogger(f"navsense.session.{self.session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._detection_stream = None
        self._event_stream = None

        if self.enabled:
            self._initialize_files()

    def _initialize_files(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.session_dir / "session.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._event_stream = open(self.session_dir / "events.jsonl", "a", encoding="utf-8")
        self._detection_stream = open(self.session_dir / "detections.jsonl", "a", encoding="utf-8")

        self.logger.info("Session logger initialized at %s", self.session_dir)

    def close(self) -> None:
        for stream in (self._event_stream, self._detection_stream):
            if stream:
                stream.close()
        self._event_stream = None
        self._detection_stream = None

        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def _write_jsonl(self, stream, payload: Dict[str, Any]) -> None:
        if not self.enabled or stream is None:
            return
        stream.write(json.dumps(payload, ensure_ascii=True) + "\n")
        stream.flush()

    def log_event(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            **payload,
        }
        self._write_jsonl(self._event_stream, record)
        summary = payload.get("message") or payload.get("text") or event_type
        self.logger.info("[%s] %s", event_type, summary)

    def log_detection_frame(self, frame_index: int, mode: str, detections: List[Dict[str, Any]]) -> None:
        if not self.enabled or not self.save_detections:
            return

        compact = []
        for det in detections:
            compact.append(
                {
                    "track_id": det.get("track_id"),
                    "class_name": det.get("class_name"),
                    "raw_class_name": det.get("raw_class_name", det.get("class_name")),
                    "confidence": round(float(det.get("confidence", 0.0)), 3),
                    "distance": round(float(det.get("distance", 0.0)), 2),
                    "zone": det.get("zone"),
                    "direction_text": det.get("direction_text"),
                    "bbox": det.get("bbox"),
                }
            )

        self._write_jsonl(
            self._detection_stream,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_index": frame_index,
                "mode": mode,
                "count": len(compact),
                "detections": compact,
            },
        )

    def log_voice_command(self, text: str, mode: str | None = None) -> None:
        if not self.enabled or not self.save_voice_commands:
            return
        self.log_event("voice_command", text=text, mode=mode)

    def log_response(self, text: str, mode: str | None = None, source: str = "system", priority: bool = False) -> None:
        self.log_event("response", text=text, mode=mode, source=source, priority=priority)

    def log_alert(self, alert: Dict[str, Any], mode: str | None = None) -> None:
        if not self.enabled or not self.save_alerts:
            return
        payload = dict(alert)
        payload["mode"] = mode
        self.log_event("alert", **payload)

