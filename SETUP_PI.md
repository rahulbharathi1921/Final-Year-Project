# NavSense Raspberry Pi 5 Full Setup Guide

Complete step-by-step guide: flash Raspberry Pi OS to running NavSense with VSCode, camera, ultrasonic sensor, and IMU.

---

## Table of Contents

1. [What You Need](#1-what-you-need)
2. [Flash Raspberry Pi OS](#2-flash-raspberry-pi-os)
3. [Initial Pi Setup (headless)](#3-initial-pi-setup-headless)
4. [Enable Hardware Interfaces](#4-enable-hardware-interfaces)
5. [Install System Packages](#5-install-system-packages)
6. [Install VSCode](#6-install-vscode)
7. [Clone the Project](#7-clone-the-project)
8. [Setup Virtual Environment](#8-setup-virtual-environment)
9. [Install Python Dependencies](#9-install-python-dependencies)
10. [Configure Gemini API Key](#10-configure-gemini-api-key)
11. [Verify Camera](#11-verify-camera)
12. [Verify I2C Devices (IMU)](#12-verify-i2c-devices-imu)
13. [Verify Ultrasonic Sensor](#13-verify-ultrasonic-sensor)
14. [Download YOLO Model](#14-download-yolo-model)
15. [Run NavSense](#15-run-navsense)
16. [Access Pi via VSCode SSH](#16-access-pi-via-vscode-ssh)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What You Need

### Hardware
- Raspberry Pi 5 (4GB recommended, 8GB also works)
- 32GB+ microSD card (Class A2 or better)
- Pi Camera Module v1 (or v2 with minor tweaks)
- HC-SR04 ultrasonic sensor
- MPU6050 IMU module
- 5V 3A USB-C power supply (Pi 5 requires official PoE+ or 27W adapter)
- Voltage divider for HC-SR04 ECHO pin (1kΩ + 2kΩ resistors)
- Jumper wires, breadboard
- USB microphone (for voice commands)
- Speaker with 3.5mm jack (or USB speaker)
- Laptop or desktop PC (for VSCode SSH)

### Software on Your Laptop
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [VSCode](https://code.visualstudio.com/) with Remote - SSH extension
- [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) or Windows Terminal (for SSH)

---

## 2. Flash Raspberry Pi OS

### Step 2.1: Download Raspberry Pi Imager

Go to [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/) and download for your OS (Windows/macOS/Linux).

### Step 2.2: Insert SD Card

Insert your microSD card into your laptop card reader.

### Step 2.3: Open Raspberry Pi Imager

Launch the Imager application.

### Step 2.4: Select OS

Click **Choose OS** and select:
- **Raspberry Pi OS (64-bit)** — recommended
- Or **Raspberry Pi OS Lite (64-bit)** if you want a minimal install without desktop

### Step 2.5: Select Storage

Click **Choose Storage** and select your SD card. **WARNING: This will ERASE the SD card. Double-check you selected the correct drive.**

### Step 2.6: Configure OS (Gear Icon)

Click the **gear icon** (⚙️) to open advanced options. Fill in:

**General tab:**
- Set hostname: `navsense` (so you can reach it as `navsense.local`)
- Enable "Set username and password": `pi` / `navsense123` (change this password!)
- Configure your WiFi: select your network, enter password
- Set locale settings: timezone `Asia/Kolkata` (or your zone), keyboard layout

**Services tab:**
- Enable SSH with password authentication
- Disable "Use password for authentication over SSH" if you prefer keys (optional)

**Click Save.**

### Step 2.7: Write the Image

Click **Write**. Confirm the erasure warning. Wait 10-15 minutes for the image to write and verify.

### Step 2.8: Boot the Pi

- Eject the SD card safely
- Insert into Pi 5
- Connect power, Ethernet (optional — WiFi is configured), keyboard/mouse (optional), HDMI
- Power on

---

## 3. Initial Pi Setup (headless)

If you have a monitor/keyboard, skip to Step 3.4.

### Step 3.1: Find Pi on Network

Wait 1-2 minutes for Pi to boot. From your laptop, open terminal/command prompt:

```bash
# On Linux/macOS - install avahi-utils first if needed
ping navsense.local

# On Windows - try
ping navsense.local
```

If `navsense.local` doesn't work, use your router to find the Pi's IP, or scan:

```bash
# On Linux/macOS
sudo nmap -sn 192.168.1.0/24 | grep -i raspberry

# On Windows, use Angry IP Scanner or Advanced IP Scanner
```

### Step 3.2: SSH into Pi

```bash
ssh pi@navsense.local
```

Or using IP:
```bash
ssh pi@<PI_IP_ADDRESS>
```

First connection: type `yes` to accept host key. Enter password you set in Step 2.6.

If connection refused and you're using WiFi, check the headless setup — sometimes SSH file needs to be created on the SD card before first boot.

### Step 3.3: Create SSH File (If SSH Fails on First Boot)

Eject SD card, mount it on laptop. Create empty file `ssh` (no extension) in the boot partition:

**Linux/macOS:**
```bash
# Mount boot partition (usually /dev/sdX1)
sudo mount /dev/sdX1 /mnt
sudo touch /mnt/ssh
sudo umount /mnt
```

**Windows (PowerShell as Admin):**
```powershell
# Find boot drive letter (e.g., D:)
New-Item -Path "D:\ssh" -ItemType File
```

Then boot Pi again.

### Step 3.4: Initial System Update

Once SSH'd in (or on keyboard/display):

```bash
# Update package lists
sudo apt update

# Full system upgrade
sudo apt full-upgrade -y

# Install common tools (if not already present)
sudo apt install -y curl wget git nano

# Reboot to apply kernel updates
sudo reboot
```

After reboot (30-60 seconds), reconnect:
```bash
ssh pi@navsense.local
```

---

## 4. Enable Hardware Interfaces

### Step 4.1: Open Raspberry Pi Configuration

```bash
sudo raspi-config
```

Navigate with arrow keys.

### Step 4.2: Enable Camera

- Go to **Interface Options**
- Select **Camera** → **Yes** (enable legacy camera)
- Note: On Bookworm, the legacy camera stack is being phased out. Picamera2 (libcamera) is the future. You don't strictly need legacy camera enabled, but it doesn't hurt.

### Step 4.3: Enable I2C

- Go to **Interface Options**
- Select **I2C** → **Yes** (enable I2C)

### Step 4.4: Enable SPI (optional, not needed for NavSense)

Skip unless you have other sensors requiring SPI.

### Step 4.5: Enable SSH (if not already)

- Go to **Interface Options**
- Select **SSH** → **Yes**

### Step 4.6: Finish and Reboot

Select **Finish** → **Yes** to reboot.

---

## 5. Install System Packages

These are OS-level dependencies required by NavSense.

```bash
sudo apt update
```

### Camera and video packages
```bash
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    libcamera-dev
```

### Audio packages
```bash
sudo apt install -y \
    portaudio19-dev \
    espeak \
    pulseaudio \
    pavucontrol \
    alsa-utils \
    libasound2-dev
```

### I2C and sensor packages
```bash
sudo apt install -y \
    i2c-tools \
    python3-smbus \
    smbus2
```

### Math and compilation tools
```bash
sudo apt install -y \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran
```

### Networking and utilities
```bash
sudo apt install -y \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev
```

### Optional: Remote desktop (for GUI access)
```bash
sudo apt install -y \
    realvnc-vnc-server \
    realvnc-vnc-viewer
```

Enable VNC:
```bash
sudo raspi-config
# Interface Options → VNC → Yes
```

---

## 6. Install VSCode

### Step 6.1: Download VSCode Server for Pi

You don't install VSCode directly on the Pi. You install a lightweight VSCode **server** on the Pi and connect from VSCode on your laptop via SSH.

On your laptop (NOT on the Pi):

### Step 6.2: Install VSCode on Your Laptop

Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)

Install the **Remote - SSH** extension:

1. Open VSCode
2. Press `Ctrl+Shift+X` to open Extensions
3. Search for `Remote - SSH` (Microsoft)
4. Click **Install**

Also install:
- `Python` (Microsoft)
- `Pylance` (optional)

### Step 6.3: Connect to Pi

1. In VSCode, press `F1` or `Ctrl+Shift+P`
2. Type `Remote-SSH: Connect to Host`
3. Select `Add New SSH Host`
4. Enter: `pi@navsense.local` (or `pi@<IP>`)
5. Select your SSH config file (default: `~/.ssh/config`)
6. Click **Connect** — VSCode will install the server on the Pi automatically
7. Enter your password when prompted

This gives you a full VSCode editor running on the Pi with syntax highlighting, terminal access, and all extensions.

---

## 7. Clone the Project

### Step 7.1: Generate SSH Key on Pi (recommended)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "pi@navsense" -f ~/.ssh/id_ed25519 -N ""

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Display public key (copy this)
cat ~/.ssh/id_ed25519.pub
```

### Step 7.2: Add SSH Key to GitHub

1. Go to [https://github.com/settings/keys](https://github.com/settings/keys)
2. Click **New SSH Key**
3. Paste the public key from Step 7.1
4. Give it a title like "NavSense Pi"

### Step 7.3: Clone the Repository

```bash
# Navigate to home directory
cd ~

# Clone the repo
git clone git@github.com:rahulbharathi1921/Final-Year-Project.git

# Navigate into the project
cd Final-Year-Project/Nav-Sense

# Or if the repo IS Nav-Sense directly
# git clone git@github.com:rahulbharathi1921/Final-Year-Project.git Nav-Sense
```

Verify:
```bash
ls -la
# You should see: main.py, config/, core/, requirements*.txt, etc.
```

### Step 7.4: Verify git remotes

```bash
git remote -v
# Should show:
# origin  git@github.com:rahulbharathi1921/Final-Year-Project.git (fetch)
# origin  git@github.com:rahulbharathi1921/Final-Year-Project.git (push)
```

---

## 8. Setup Virtual Environment

### Step 8.1: Create Virtual Environment

```bash
cd ~/Final-Year-Project/Nav-Sense

# Create venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify activation
python --version
# Should show Python 3.11.x or similar
```

### Step 8.2: Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

### Step 8.3: (Optional) Set venv to use system site packages for picamera2

If picamera2 gives import issues, you may need:
```bash
# Recreate venv with system site packages
deactivate
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

---

## 9. Install Python Dependencies

### Step 9.1: Install Shared Requirements

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 9.2: Install Pi-Specific Requirements

```bash
pip install -r requirements-pi.txt
```

### Step 9.3: Install picamera2 explicitly

```bash
pip install picamera2
```

### Step 9.4: Install ultralytics (YOLO)

```bash
pip install ultralytics
```

### Step 9.5: Test Key Imports

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics OK')"
python -c "from picamera2 import Picamera2; print('Picamera2 OK')"
python -c "import faster_whisper; print('Faster Whisper OK')"
python -c "import gpiozero; print('gpiozero OK')"
python -c "import smbus; print('smbus OK')"
python -c "import pyttsx3; print('pyttsx3 OK')"
python -c "import gTTS; print('gTTS OK')"
python -c "import pygame; print('pygame OK')"
```

All should print without errors.

---

## 10. Configure Gemini API Key

### Step 10.1: Get Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Click **Create API Key**
4. Copy the key

### Step 10.2: Create .env File

```bash
cd ~/Final-Year-Project/Nav-Sense

# Create .env file
nano .env
```

Add:
```env
GEMINI_API_KEY="AIzaSy....................."
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### Step 10.3: Verify .env Loads

```bash
# The app uses python-dotenv, but test it manually
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GEMINI_API_KEY:', 'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET')"
```

### Step 10.4: (Alternative) Export as Environment Variable

```bash
export GEMINI_API_KEY="AIzaSy....................."
```

To make this permanent:
```bash
echo 'export GEMINI_API_KEY="AIzaSy....................."' >> ~/.bashrc
source ~/.bashrc
```

---

## 11. Verify Camera

### Step 11.1: Check libcamera Detects Camera

```bash
# List cameras
libcamera-hello --list-cameras
```

You should see:
```
Available cameras
-----------------
0 : arducam-pivariety [4640x3472] (/base/axi/pcie@100000000/pci@0,0/axi/pcie@9,0/vcm@0)
   Formats: 'SRGGB10_CSI2P' -> 'BGR888', 'RGB888' ...
```

If no cameras found, check ribbon cable connection and camera enablement.

### Step 11.2: Quick Camera Preview (5 seconds)

```bash
libcamera-hello
```

A window should appear showing camera feed for 5 seconds. If running headless (no display), this will fail — that's OK.

### Step 11.3: Test Capture

```bash
libcamera-still -o test.jpg
```

Check if `test.jpg` was created:
```bash
ls -lh test.jpg
```

### Step 11.4: Test with Python picamera2

```bash
python3 -c "
from picamera2 import Picamera2
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={'size': (640, 480)}))
picam.start()
import time
time.sleep(2)
picam.stop()
print('Camera test PASSED')
"
```

### Step 11.5: Test NavSense Camera Handler Directly

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from core.camera_handler import CameraHandler
import yaml

with open('config/settings.pi.yaml') as f:
    config = yaml.safe_load(f)

cam = CameraHandler(config['camera'])
cam.platform = 'raspberry_pi'

if cam.initialize():
    print(f'Camera OK: {cam.width}x{cam.height} @ {cam.fps_target}fps')
    frame = cam.get_frame()
    print(f'Frame shape: {frame.shape if frame is not None else None}')
    cam.release()
else:
    print('Camera FAILED')
"
```

---

## 12. Verify I2C Devices (IMU)

### Step 12.1: Check I2C Bus

```bash
# Install i2c-tools if not already
sudo apt install -y i2c-tools

# Scan I2C bus 1
sudo i2cdetect -y 1
```

You should see something like:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- 68 -- -- -- --
70: -- -- -- -- -- -- -- --
```

Address `0x68` is the MPU6050. If you see `--` instead, check wiring.

### Step 12.2: Verify IMU with Python

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from core.imu_sensor import MPU6050Sensor

config = {
    'enabled': True,
    'bus': 1,
    'address': '0x68'
}
imu = MPU6050Sensor(config)

if imu.initialize():
    reading = imu.read()
    if reading:
        print(f'IMU OK - Pitch: {reading[\"pitch_deg\"]:.1f}°, Roll: {reading[\"roll_deg\"]:.1f}°')
        print(f'Accel: X={reading[\"accel\"][\"x\"]:.3f}, Y={reading[\"accel\"][\"y\"]:.3f}, Z={reading[\"accel\"][\"z\"]:.3f}')
    else:
        print('IMU read returned None')
else:
    print('IMU init FAILED')
"
```

### Step 12.3: Wiring Check for MPU6050

Double-check these connections:
| MPU6050 | Pi 5 |
|---------|------|
| VCC | 3.3V (pin 1) |
| GND | GND (pin 6) |
| SDA | GPIO2 / SDA (pin 3) |
| SCL | GPIO3 / SCL (pin 5) |

---

## 13. Verify Ultrasonic Sensor

### Step 13.1: Wiring for HC-SR04

| HC-SR04 | Pi 5 |
|---------|------|
| VCC | 5V (pin 2 or 4) |
| GND | GND (pin 6) |
| TRIG | GPIO23 (pin 16) |
| ECHO | GPIO24 (pin 18) via voltage divider |

**IMPORTANT: The ECHO pin outputs 5V which will damage Pi GPIO (3.3V max). You MUST use a voltage divider.**

Voltage divider: ECHO → 1kΩ resistor → GPIO24 AND 2kΩ resistor → GND. This divides 5V to ~3.3V.

### Step 13.2: Test with Python

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from core.ultrasonic_sensor import UltrasonicSensor

config = {
    'enabled': True,
    'trigger_pin': 23,
    'echo_pin': 24,
    'max_distance_m': 3.0,
    'sample_count': 3,
    'offset_m': 0.0
}
sensor = UltrasonicSensor(config)

if sensor.initialize():
    print('Ultrasonic init OK')
    for i in range(5):
        result = sensor.read()
        if result:
            print(f'Distance: {result[\"distance_m\"]*100:.1f} cm (samples: {result[\"samples\"]})')
        else:
            print('No reading')
        import time
        time.sleep(0.5)
    sensor.shutdown()
else:
    print('Ultrasonic init FAILED')
"
```

Expected output: distance readings in cm (e.g., `45.2 cm`, `120.5 cm`). Returns `None` or very high values if no obstacle in range.

### Step 13.3: Manual GPIO Test (optional)

```bash
python3 -c "
import gpiozero
from gpiozero import DistanceSensor
sensor = DistanceSensor(echo=24, trigger=23, max_distance=3.0)
print(f'Distance: {sensor.distance * 100:.1f} cm')
sensor.close()
"
```

---

## 14. Download YOLO Model

### Step 14.1: Check Existing Models

```bash
ls -lh models/
```

If `yolov5n.pt` or `yolov5s.pt` already exists, skip to Step 14.4.

### Step 14.2: Create Models Directory

```bash
mkdir -p models
cd models
```

### Step 14.3: Download YOLOv5 Nano

```bash
# Download from Ultralytics GitHub releases
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt

# Or download small model (more accurate, slightly heavier)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### Step 14.4: Verify Model

```bash
ls -lh models/
python3 -c "
from ultralytics import YOLO
model = YOLO('models/yolov5s.pt')
print(f'Model loaded: {len(model.names)} classes')
print('Classes:', list(model.names.values())[:10])
"
```

---

## 15. Run NavSense

### Step 15.1: Test with Laptop Config First (if you have monitor)

```bash
cd ~/Final-Year-Project/Nav-Sense
source .venv/bin/activate

python main.py --config config/settings.laptop.yaml
```

### Step 15.2: Run with Pi Config

```bash
cd ~/Final-Year-Project/Nav-Sense
source .venv/bin/activate

python main.py --config config/settings.pi.yaml
```

Expected output:
```
[NavSense] Loading configuration from config/settings.pi.yaml...
[NavSense] Configuration loaded successfully

============================================================
NavSense - Navigation Assistant for Visually Impaired
============================================================

[1/7] Initializing camera...
[CameraHandler] Initializing camera 0...
[CameraHandler] Picamera2 backend initialized
[CameraHandler] Camera initialized: 640x480

[2/5] Loading object detector...
[Detector] Loading YOLO model from models/yolov5s.pt...
[Detector] Using device: cpu
[Detector] Model loaded successfully!

[3/5] Initializing distance & direction calculator...
[3/5] Calculator initialized

[4/5] Initializing alert system...
[4/5] Alert system initialized

[5/7] Initializing voice interface...
[VoiceHandler] Loading Whisper 'tiny.en' (Offline)...
[VoiceHandler] System initialized successfully!

[6/7] Initializing LLM (Gemini API)...
[LLMHandler] Initialized with model: gemini / gemini-2.5-flash-lite

[7/7] Initializing object tracker...
[7/7] Tracker initialized
[Sensors] Ultrasonic sensor ready
[Sensors] MPU6050 sensor ready

============================================================
All components initialized successfully!
============================================================

Hello Joe! NavSense is active and 100 percent offline. Which mode do you want: indoor, outdoor, or jarvis?
[NavSense] Starting voice recognition...
```

### Step 15.3: If Running Headless (No Display)

The Pi config already has `show_window: false` so display won't be needed. Voice-only mode works.

### Step 15.4: Check Logs

```bash
# See latest session log
ls -lt logs/
tail -f logs/session_*/session.log
```

---

## 16. Access Pi via VSCode SSH

### Step 16.1: From VSCode (on your laptop)

1. Open VSCode
2. `F1` → `Remote-SSH: Connect to Host` → `navsense.local`
3. VSCode opens a new window, installs server on Pi
4. Open folder: `~/Final-Year-Project/Nav-Sense`

### Step 16.2: Recommended VSCode Extensions for Pi

Install these on the Pi side (after SSH connect):
- **Python** (Microsoft)
- **Pylance** (optional)
- **GitLens** (optional)
- **Even Better TOML** (for YAML/TOML editing)

### Step 16.3: Set Python Interpreter

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Choose: `~/Final-Year-Project/Nav-Sense/.venv/bin/python`

### Step 16.4: Git Integration in VSCode

1. `Ctrl+Shift+G` to open Source Control panel
2. Sign in to GitHub: `F1` → `Git: Sign In`
3. You can now commit, push, pull directly from VSCode

---

## 17. Troubleshooting

### Camera Issues

**Problem: `picamera2` import fails**
```bash
sudo apt install -y python3-picamera2 python3-libcamera
pip install picamera2 --force-reinstall
```

**Problem: No camera detected**
```bash
# Check camera ribbon cable seating
# Try reconnecting both ends
libcamera-hello --list-cameras
# If still missing: check /boot/config.txt has `dtparam=cam1_on` or similar
sudo nano /boot/firmware/config.txt
# Look for: dtparam=icm
# Add if missing: dtoverlay=imx219
```

**Problem: Camera works but shows dark/purple image**
- Common with v1 camera in bright light — try adjusting
- Could be focus issue with NoIR camera
- Ribbon cable quality — try replacement

### Audio Issues

**Problem: pyttsx3 not speaking**
```bash
# Test espeak directly
espeak "Hello, this is a test" -ven
# If no audio output, install and configure:
sudo apt install -y espeak
```

**Problem: Faster Whisper not loading**
```bash
# Force reinstall
pip install faster-whisper --force-reinstall --no-cache-dir
```

**Problem: gTTS fails with network error**
- gTTS requires internet. If offline, ensure pyttsx3 fallback is working.
- Check: the code automatically falls back to pyttsx3 if gTTS fails

**Problem: USB microphone not detected**
```bash
# List audio devices
pactl list sources short
# or
arecord -l
# Set default in pulseaudio:
pavucontrol
```

### Sensor Issues

**Problem: IMU not on I2C bus**
```bash
# Check wiring: SDA→GPIO2, SCL→GPIO3
# Try bus 0 instead of bus 1:
sudo i2cdetect -y 0
# If found on bus 0, change config: bus: 0
```

**Problem: Ultrasonic returns max distance always**
- Check voltage divider — without it, GPIO might not read correctly
- Check ground connection
- Test with object 10-50cm away

**Problem: Ultrasonic returns 0 or very small values**
- Check TRIG/ECHO pins match config (BCM 23/24)
- Try a different HC-SR04 unit

### Python/Pip Issues

**Problem: pip install hangs or fails**
```bash
# Clear pip cache
pip cache purge
# Upgrade pip
pip install --upgrade pip
# Try again with verbose
pip install -r requirements.txt -v
```

**Problem: version 'GLIBC_2.34' not found**
- Some prebuilt wheels need newer GLIBC. Rebuild from source:
```bash
pip install --no-binary :all: numpy
```

### Git Issues

**Problem: Git clone fails with "Connection refused"**
```bash
# Test SSH to GitHub
ssh -T git@github.com
# If it fails, your SSH key might not be added to GitHub
```

**Problem: Git push asks for password every time**
```bash
# Configure SSH to use the key automatically
nano ~/.ssh/config
```
Add:
```
Host github.com
    User git
    Hostname github.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/id_ed25519
```

### Network Issues

**Problem: Can't reach navsense.local**
```bash
# From laptop, install avahi on Linux
sudo apt install -y avahi-discover

# On Windows, install Bonjour Services or use Nmap
# Or find IP from router
arp -a | grep -i rasp
```

**Problem: WiFi keeps disconnecting**
```bash
# Check WiFi signal
iwconfig wlan0
# Or use:
iw dev wlan0 link

# Set WiFi country for proper channels
sudo raspi-config
# Localisation Options → WLAN Country → IN (or your country)
```

### Startup Issues

**Problem: NavSense starts but immediately crashes**
```bash
# Check Python version
python --version  # Should be 3.9+

# Check all imports
python -c "import sys; sys.path.insert(0, '.'); from core import *"

# Check config loads
python -c "
import yaml
with open('config/settings.pi.yaml') as f:
    c = yaml.safe_load(f)
print('Config keys:', list(c.keys()))
"

# Run with full traceback
python main.py --config config/settings.pi.yaml 2>&1
```

---

## Quick Reference Commands

### Essential everyday commands on Pi:

```bash
# Navigate to project
cd ~/Final-Year-Project/Nav-Sense

# Activate venv
source .venv/bin/activate

# Run NavSense
python main.py --config config/settings.pi.yaml

# Check logs
tail -f logs/session_*/session.log

# Git pull latest
git pull

# Check system resources
htop
# or
top

# Check GPU/memory
vcgencmd measure_temp
free -h

# Reboot Pi
sudo reboot

# Shutdown Pi
sudo shutdown now

# Check last 50 lines of session log
tail -50 logs/session_*/session.log
```

### Pinout Reference

```
    3.3V  (1) (2)  5V
   GPIO2  (3) (4)  5V
   GPIO3  (5) (6)  GND
   GPIO4  (7) (8)  GPIO14 (TXD)
     GND  (9) (10) GPIO15 (RXD)
  GPIO17 (11) (12) GPIO18
  GPIO27 (13) (14) GND
  GPIO22 (15) (16) GPIO23 --- TRIG (HC-SR04)
    3.3V (17) (18) GPIO24 --- ECHO (HC-SR04 via voltage divider)
  GPIO10 (19) (20) GND
   GPIO9 (21) (22) GPIO25
  GPIO11 (23) (24) GPIO8
     GND (25) (26) GPIO7
   ID_SD (27) (28) ID_SC
   GPIO5 (29) (30) GND
   GPIO6 (31) (32) GPIO12
  GPIO13 (33) (34) GND
  GPIO19 (35) (36) GPIO16
  GPIO26 (37) (38) GPIO20
     GND (39) (40) GPIO21
```

MPU6050: VCC→3.3V(1), GND→GND(6), SDA→GPIO2(3), SCL→GPIO3(5)