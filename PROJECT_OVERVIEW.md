# NavSense Project Overview

This document is a future-facing overview for engineers and AI agents working on this repository.

## Purpose

NavSense is an offline-first navigation assistant for visually impaired users. Its main goals are:
- detect nearby objects
- estimate direction and approximate distance
- provide spoken guidance
- warn about obstacles and hazards
- support follow-up voice queries and a lightweight Jarvis-style knowledge mode

The repository currently supports:
- laptop mode
- Raspberry Pi 5 mode

## Entry Point

- Main runtime: [main.py](./main.py)

`main.py` owns:
- configuration loading
- component initialization
- main camera/detection loop
- voice command routing
- safety guidance logic
- tracking state
- sensor fusion usage

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
  - laptop: OpenCV camera backend
  - Pi: Picamera2 backend

- [core/detector.py](./core/detector.py)
  - YOLO object detection
  - current model path defaults to `models/yolov5n.pt`

- [core/distance_direction.py](./core/distance_direction.py)
  - converts detections into:
    - approximate distance
    - left / center / right labels

### Tracking
- [core/object_tracker.py](./core/object_tracker.py)
  - object ID persistence
  - class smoothing
  - current implementation uses global assignment with distance + IoU + class penalty

### Voice
- [core/voice_handler.py](./core/voice_handler.py)
  - Whisper-based STT
  - pyttsx3 / gTTS TTS handling
  - self-echo suppression for speaker bleed

### Alerts
- [core/alert_system.py](./core/alert_system.py)
  - priority scoring
  - cooldown
  - spoken warning generation

### LLM
- [core/llm_handler.py](./core/llm_handler.py)
  - Gemini API integration only
  - current default model: `gemini-2.5-flash-lite`

### Raspberry Pi Sensors
- [core/ultrasonic_sensor.py](./core/ultrasonic_sensor.py)
  - HC-SR04 forward distance

- [core/imu_sensor.py](./core/imu_sensor.py)
  - MPU6050 pitch / roll context

- [core/sensor_fusion.py](./core/sensor_fusion.py)
  - merges ultrasonic + IMU into navigation context
  - ultrasonic is treated as authoritative for close center obstacles

## Navigation Logic

The navigation layer should stay deterministic.

Rule:
- mobility and safety questions must be answered from sensors + detections
- broad knowledge questions may use Gemini

This split is intentional. Do not move path-safety decisions into the LLM.

Current safety signals:
- camera detections
- estimated object distance
- zone hazard scoring
- ultrasonic straight-ahead obstacle override

## Raspberry Pi Hardware Model

Assumed hardware for current Pi profile:
- Pi Camera Module v1: centered
- HC-SR04: centered with camera
- MPU6050: slightly down-left of camera

Interpretation:
- camera and ultrasonic share the forward lane
- MPU6050 is orientation reference only
- IMU physical offset should not be treated as obstacle position

## Dependency Layout

Shared:
- [requirements.txt](./requirements.txt)

Laptop:
- [requirements-laptop.txt](./requirements-laptop.txt)

Pi:
- [requirements-pi.txt](./requirements-pi.txt)

Pi-specific OS bring-up:
- [RASPBERRY_PI_SETUP.md](./RASPBERRY_PI_SETUP.md)

## Current Stability Notes

### Stable enough
- laptop camera path
- voice routing structure
- Gemini integration
- YOLOv5-based object pipeline
- basic Pi sensor integration structure

### Needs real-device validation
- Picamera2 runtime on your exact Pi image
- HC-SR04 GPIO timing in your wiring
- MPU6050 orientation sign conventions
- Pi 5 performance under simultaneous:
  - camera
  - YOLO
  - Whisper
  - voice output

## Recommended Future Work

### High priority
1. Add hardware smoke tests:
   - camera-only
   - ultrasonic-only
   - MPU6050-only

2. Add focal length calibration flow.

3. Add ultrasonic offset calibration flow.

4. Add explicit Pi performance profile:
   - lower FPS
   - lower detection frequency
   - `tiny.en` STT default

### Medium priority
1. Add tilt-aware warnings from MPU6050.
2. Add obstacle memory from ultrasonic when camera misses near obstacles.
3. Add a dedicated `scripts/` folder for platform bring-up tests.

### Low priority
1. Split voice and sensor logs into separate channels.
2. Add structured telemetry dashboards for field testing.

## Rules For Future AI Edits

If an AI agent changes this repo:
- do not break laptop default flow
- keep `config/settings.yaml` as a safe default
- preserve dual-mode compatibility
- prefer additive Pi support over invasive rewrites
- never route safety-critical motion guidance through the LLM
- verify imports and config keys after cross-file changes

If changing config behavior:
- support explicit config selection
- do not force manual file edits just to switch platforms

If changing sensor logic:
- ultrasonic should remain the primary forward close-range obstacle sensor
- IMU should remain orientation context, not obstacle localization
