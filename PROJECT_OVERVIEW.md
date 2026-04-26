# NavSense Project Overview

This document is a future-facing overview for engineers and AI agents working on this repository.

## Purpose

NavSense is an offline-first navigation assistant for visually impaired users. Its main goals are:
- detect nearby objects
- estimate direction and approximate distance
- provide spoken guidance
- warn about obstacles and hazards
- support follow-up voice queries and a lightweight Jarvis-style knowledge mode

The repository supports:
- laptop mode
- Raspberry Pi 5 mode

## Entry Point

- Main runtime: [main.py](./main.py)

`main.py` owns:
- configuration loading (YAML + JSON)
- component initialization (camera, detector, calculator, alert, voice, LLM, tracker, sensors)
- main camera/detection loop
- voice command routing with fuzzy matching
- safety guidance logic (zone hazard scoring)
- object tracking state
- sensor fusion usage (ultrasonic + IMU)
- session logging (JSONL + human-readable)

## Runtime Profiles

Config files:
- [config/settings.yaml](./config/settings.yaml): safe default, currently laptop-oriented
- [config/settings.laptop.yaml](./config/settings.laptop.yaml): explicit laptop profile
- [config/settings.pi.yaml](./config/settings.pi.yaml): explicit Raspberry Pi 5 profile

Config selection order:
1. `--config <path>` command-line argument
2. `NAVSENSE_CONFIG` environment variable
3. default `config/settings.yaml`

Examples:

Laptop:
```bash
python main.py --config config/settings.laptop.yaml
```

Raspberry Pi:
```bash
python main.py --config config/settings.pi.yaml
```

Or:
```bash
export NAVSENSE_CONFIG=config/settings.pi.yaml
python main.py
```

## Core Modules

### Vision
- [core/camera_handler.py](./core/camera_handler.py)
  - laptop: OpenCV VideoCapture backend
  - Pi: Picamera2 backend with threaded capture
  - Platform-aware backend selection via `platform` config key
  - FOV calculation and FPS tracking

- [core/detector.py](./core/detector.py)
  - YOLO object detection
  - Primary backend: `ultralytics` YOLO (from `ultralytics` pip package)
  - Fallback: PyTorch Hub YOLOv5 (for custom .pt weights)
  - Default model path: `models/yolov5s.pt`
  - Supports mode-based detection filtering via `filter_by_mode()`
  - Config keys: `confidence_threshold`, `iou_threshold`, `max_detections`

- [core/distance_direction.py](./core/distance_direction.py)
  - Converts detections into: approximate distance (meters), direction (left/center/right)
  - Distance formula: `Distance = (Real_Size × Focal_Length) / Pixel_Size`
  - Default focal_length: 615.0 (calibrated for 640x480)
  - Center zone threshold: 22% of frame width
  - Fallback heuristic for unknown objects based on bbox area ratio

### Tracking
- [core/object_tracker.py](./core/object_tracker.py)
  - Object ID persistence across frames
  - Centroid matching with Hungarian algorithm (linear_sum_assignment)
  - IoU + centroid distance + class penalty cost matrix
  - Class-vote smoothing to prevent one-frame misclassifications
  - Config: `max_disappeared=5`, `max_distance=120.0` pixels

### Voice
- [core/voice_handler.py](./core/voice_handler.py)
  - STT: `faster_whisper` (CPU, int8) — offline, sub-second
  - TTS: gTTS (online, higher quality) with pyttsx3 fallback (offline)
  - Self-echo suppression: RMS threshold + text similarity + assistant-speech detection
  - Barge-in: high RMS during TTS interrupts speech
  - Interrupt commands: stop, quiet, pause, resume, mute alerts, repeat
  - Keyword whitelist filtering for command detection
  - Pygame for gTTS audio playback

### Alerts
- [core/alert_system.py](./core/alert_system.py)
  - Priority scoring: high (100), medium (50), low (20) + distance bonus
  - Cooldown per object-type + zone combination
  - Danger categories: immediate, high, medium, low (from object_data.json)
  - Auto-speak rules: high always, medium center-only, low query-only

### LLM
- [core/llm_handler.py](./core/llm_handler.py)
  - Gemini API only
  - Default model: `gemini-2.5-flash-lite`
  - 5-second response cache (TTL)
  - Response sanitization: strips echoed prompts, metadata, repetitions
  - Max tokens: 80, Temperature: 0.2
  - Scene description + query answering

### Pi Sensors
- [core/ultrasonic_sensor.py](./core/ultrasonic_sensor.py)
  - HC-SR04 via gpiozero `DistanceSensor`
  - BCM pin config (default: TRIG=23, ECHO=24)
  - Median filtering across `sample_count` reads
  - Config: `offset_m` for calibration, `max_distance_m`

- [core/imu_sensor.py](./core/imu_sensor.py)
  - MPU6050 via smbus2 I2C
  - I2C bus 1, address 0x68 (default)
  - Accelerometer + gyroscope reads
  - Pitch/roll from atan2 acceleration

- [core/sensor_fusion.py](./core/sensor_fusion.py)
  - Merges ultrasonic + IMU into navigation context
  - `augment_detections()`: attaches `forward_distance_m`, `imu_pitch_deg`, `imu_roll_deg` to each detection
  - Ultrasonic treated as authoritative for close center obstacles
  - `forward_obstacle_distance()` returns ultrasonic reading directly

### Logging
- [core/session_logger.py](./core/session_logger.py)
  - Per-session directory: `logs/session_YYYYMMDD_HHMMSS/`
  - Files: `session.log` (human-readable), `events.jsonl`, `detections.jsonl`
  - Logged events: mode changes, voice commands, responses, alerts, detection frames

## Navigation Logic

The navigation layer is deterministic.

Rule:
- mobility and safety questions must be answered from sensors + detections
- broad knowledge questions may use Gemini

This split is intentional. Do not move path-safety decisions into the LLM.

Current safety signals:
- camera detections (distance + zone)
- zone hazard scoring (count × 1.5 + 3.5/nearest_distance, center +1.0 bonus)
- ultrasonic straight-ahead obstacle override (≥0.8m adds to hazard score)
- `forward_blocked()`: True when ultrasonic ≤ 0.8m

## Raspberry Pi Hardware Model

Assumed hardware for current Pi profile:
- Pi Camera Module v1: centered
- HC-SR04: centered with camera (shares forward lane)
- MPU6050: slightly down-left of camera (orientation reference only)

Note: IMU physical offset should not be treated as obstacle position.

## Dependency Layout

Shared:
- [requirements.txt](./requirements.txt)

Laptop:
- [requirements-laptop.txt](./requirements-laptop.txt)

Pi:
- [requirements-pi.txt](./requirements-pi.txt)

Pi-specific OS bring-up:
- [RASPBERRY_PI_SETUP.md](./RASPBERRY_PI_SETUP.md)

## Additional Config Files

- [config/object_data.json](./config/object_data.json): object real-world sizes, COCO classes, danger categories
- [config/voice_vocab.json](./config/voice_vocab.json): phonetic variations for mode names and object aliases

## Models Directory

- `models/yolov5n.pt` — nano variant (lighter, faster)
- `models/yolov5nu.pt` — nano with updated COCO classes
- `models/yolov5s.pt` — small variant (default, used by configs)

## Current Stability Notes

### Stable enough
- laptop camera path (OpenCV)
- Picamera2 runtime on Bookworm Pi OS
- voice routing structure
- Gemini integration
- YOLOv5-based object pipeline
- basic Pi sensor integration structure (ultrasonic, IMU, sensor fusion)
- session logging

### Needs real-device validation
- HC-SR04 GPIO timing with your exact wiring
- MPU6050 orientation sign conventions
- Pi 5 performance under simultaneous: camera + YOLO + Whisper + voice output
- Ultrasonic `offset_m` calibration for your mount
- Focal length calibration for your specific camera

## Recommended Future Work

### High priority
1. Add hardware smoke tests: camera-only, ultrasonic-only, MPU6050-only
2. Add focal length calibration flow
3. Add ultrasonic offset calibration flow
4. Add explicit Pi performance profile: lower FPS, tiny.en STT default

### Medium priority
1. Add tilt-aware warnings from MPU6050
2. Add obstacle memory from ultrasonic when camera misses near obstacles
3. Add a dedicated `scripts/` folder for platform bring-up tests

### Low priority
1. Split voice and sensor logs into separate channels
2. Add structured telemetry dashboards for field testing

## Rules For Future AI Edits

If an AI agent changes this repo:
- do not break laptop default flow
- keep `config/settings.yaml` as a safe default
- preserve dual-mode compatibility
- prefer additive Pi support over invasive rewrites
- never route safety-critical motion guidance through the LLM
- verify imports and config keys after cross-file changes

If changing config behavior:
- support explicit config selection via `--config` or `NAVSENSE_CONFIG` env var
- do not force manual file edits just to switch platforms

If changing sensor logic:
- ultrasonic should remain the primary forward close-range obstacle sensor
- IMU should remain orientation context, not obstacle localization

If changing detection model:
- update default `model_path` in settings.yaml and settings.laptop.yaml consistently
- the `ultralytics` YOLO backend is primary; PyTorch Hub is fallback for custom .pt files