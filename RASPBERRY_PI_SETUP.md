# NavSense Raspberry Pi 5 Setup

This guide is for:
- Raspberry Pi 5
- 4GB RAM
- Pi Camera Module v1
- HC-SR04 ultrasonic sensor
- MPU6050 IMU

Your current hardware layout assumption in the code:
- camera centered
- ultrasonic centered
- MPU6050 slightly down-left of camera

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
- `python3-picamera2` is the correct way to get Pi camera support on Raspberry Pi OS.
- `portaudio19-dev` helps `pyaudio` build cleanly.
- `espeak` is commonly used by `pyttsx3` on Linux.

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

If `pyaudio` fails, confirm `portaudio19-dev` is installed and retry.

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
- `TRIG` -> GPIO23
- `ECHO` -> GPIO24 through voltage divider

Your voltage divider on `ECHO` is correct and required.

### MPU6050
- `VCC` -> 3.3V
- `GND` -> GND
- `SDA` -> GPIO2 / SDA
- `SCL` -> GPIO3 / SCL

## 10. Switch config to Raspberry Pi mode

Edit `config/settings.yaml`:

```yaml
platform:
  target: "raspberry_pi"
  headless: true

camera:
  backend: "picamera2"

display:
  show_window: false

sensors:
  ultrasonic:
    enabled: true
  imu:
    enabled: true
```

Recommended Pi tuning:

```yaml
camera:
  width: 640
  height: 480
  fps_target: 15

voice:
  model_size: "tiny.en"
```

Reason:
- Pi 5 can run more than this, but `tiny.en` and lower FPS will give better real-time behavior.

## 11. First run

Run:

```bash
python main.py
```

## 12. If object detection is slow

Try these changes in `config/settings.yaml`:

```yaml
camera:
  fps_target: 12

detection:
  max_detections: 10

voice:
  model_size: "tiny.en"
```

If PyTorch/YOLO is still too heavy on the Pi image you use:
- test the camera + voice + sensors first
- then install a Pi-compatible PyTorch wheel for your OS

## 13. Recommended bring-up order

Do not debug everything at once.

1. camera only
2. ultrasonic only
3. MPU6050 only
4. voice only
5. full NavSense

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
- confirm I2C is enabled
- confirm module power and ground

### Ultrasonic always reads zero or max

Check:
- voltage divider on `ECHO`
- ground connection
- TRIG/ECHO pin numbers match config

### App runs but no display

This is expected in Pi headless mode:

```yaml
platform:
  headless: true
display:
  show_window: false
```

Voice-first mode is the intended Pi deployment path.

## 15. Recommended next calibration

After the first successful run:
- calibrate `distance.focal_length`
- tune `sensors.ultrasonic.offset_m`
- verify MPU6050 pitch/roll orientation
- retune voice thresholds for your mic/speaker setup
