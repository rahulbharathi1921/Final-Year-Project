# NavSense Raspberry Pi 5 Setup

This guide is for:
- Raspberry Pi 5 (4GB RAM)
- Pi Camera Module v1
- HC-SR04 ultrasonic sensor
- MPU6050 IMU

Your hardware layout assumption in the code:
- camera centered
- ultrasonic centered (shares forward lane with camera)
- MPU6050 slightly down-left of camera (orientation reference only)

## 1. Flash and update Raspberry Pi OS

Use a recent Raspberry Pi OS Bookworm image.

After first boot:
```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

## 2. Enable hardware interfaces

Run:
```bash
sudo raspi-config
```

Enable:
- `Interface Options` -> `I2C`
- `Interface Options` -> `Camera`

Then reboot:
```bash
sudo reboot
```

## 3. Install system packages

Install OS-level dependencies first:
```bash
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-dev \
  python3-picamera2 \
  python3-libcamera \
  python3-smbus \
  i2c-tools \
  portaudio19-dev \
  espeak \
  libatlas-base-dev \
  libopenblas-dev
```

Notes:
- `python3-picamera2` is the correct way to get Pi camera support on Bookworm
- `portaudio19-dev` helps `pyaudio` build cleanly
- `espeak` is used by `pyttsx3` on Linux

## 4. Clone the repo

```bash
git clone <your-repo-url>
cd Nav-Sense
```

## 5. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install Python dependencies

Install shared + Pi-specific packages:
```bash
python -m pip install -r requirements.txt -r requirements-pi.txt
```

If `pyaudio` fails, confirm `portaudio19-dev` is installed and retry:
```bash
python -m pip install pyaudio
```

Note: On Pi, `torch`/`torchvision` should be installed separately only after validating available wheels for your exact OS. The codebase uses `ultralytics` which handles its own model loading.

## 7. Configure Gemini key

Create `.env` in the repo root:
```env
GEMINI_API_KEY="your_key_here"
```

## 8. Verify camera and I2C

Check Pi camera:
```bash
libcamera-hello
```

Check I2C device list:
```bash
i2cdetect -y 1
```

You should normally see MPU6050 at `0x68`.

## 9. Wiring assumptions

### HC-SR04
- `VCC` -> 5V
- `GND` -> GND
- `TRIG` -> GPIO23 (BCM numbering)
- `ECHO` -> GPIO24 through voltage divider (required, do not skip)

### MPU6050
- `VCC` -> 3.3V
- `GND` -> GND
- `SDA` -> GPIO2 / SDA
- `SCL` -> GPIO3 / SCL

## 10. Run with Pi config profile

Do NOT edit config files manually. Use the `--config` flag:

```bash
python main.py --config config/settings.pi.yaml
```

Or set the environment variable:
```bash
export NAVSENSE_CONFIG=config/settings.pi.yaml
python main.py
```

This selects the correct settings:
- `camera.backend: "picamera2"`
- `sensors.ultrasonic.enabled: true`
- `sensors.imu.enabled: true`
- `display.show_window: false`
- `voice.model_size: "tiny.en"`
- `camera.fps_target: 15`

## 11. Recommended bring-up order

Do not debug everything at once:

1. camera only (run with display or check logs for frame output)
2. ultrasonic only (check logs for distance readings)
3. MPU6050 only (check logs for pitch/roll values)
4. voice only (test `tiny.en` model)
5. full NavSense

## 12. Tuning for your setup

If object detection is slow, adjust `settings.pi.yaml`:
```yaml
camera:
  fps_target: 12

detection:
  max_detections: 10

voice:
  model_size: "tiny.en"
```

If PyTorch/YOLO is too heavy:
- test camera + voice + sensors first
- then install a Pi-compatible PyTorch wheel if needed

## 13. Calibration steps (after first successful run)

1. Calibrate `distance.focal_length` (default 615.0) with known-distance objects
2. Tune `sensors.ultrasonic.offset_m` (default 0.0) for your mount
3. Verify MPU6050 pitch/roll orientation sign conventions
4. Retune voice RMS thresholds for your mic/speaker setup

## 14. Common issues

### `picamera2` import fails

Install:
```bash
sudo apt install -y python3-picamera2 python3-libcamera
```

### `pyaudio` build fails

Install:
```bash
sudo apt install -y portaudio19-dev
```

Then retry:
```bash
python -m pip install pyaudio
```

### IMU not detected

Check:
```bash
i2cdetect -y 1
```

If `0x68` is missing:
- recheck SDA/SCL wiring
- confirm I2C is enabled in `raspi-config`
- confirm module power and ground

### Ultrasonic always reads zero or max

Check:
- voltage divider on `ECHO` (required!)
- ground connection
- TRIG/ECHO pin numbers match config (BCM numbering: 23/24)

### App runs but no display

This is expected in Pi headless mode with `show_window: false`. Voice-first mode is the intended Pi deployment path.

Check logs for confirmation:
```bash
tail -f logs/session_*/session.log
```

## 15. Hardware layout verification

The Pi config (`settings.pi.yaml`) defines:
```yaml
sensors:
  ultrasonic:
    mount_position: "center"
  imu:
    mount_position: "down_left_of_camera"
  fusion:
    layout_note: "camera_centered_ultrasonic_centered_imu_down_left"
    primary_forward_sensor: "ultrasonic"
```

Interpretation:
- camera and ultrasonic share the forward lane
- MPU6050 is orientation reference only — its physical offset should NOT be treated as obstacle position
- ultrasonic is authoritative for forward close-range obstacle detection (≤0.8m triggers high-priority alert)