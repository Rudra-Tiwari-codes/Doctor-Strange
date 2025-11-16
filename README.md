<div align="center">

# ✨ Doctor Strange AR Filter

### *Harness the power of the Mystic Arts through Computer Vision*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="assets/doctor_strange_1.jpg" width="45%" alt="Portal Effect 1" />
<img src="assets/doctor_strange_2.jpg" width="45%" alt="Portal Effect 2" />

</div>

---

## 🎯 Features

<table>
<tr>
<td width="50%">

### 🖐️ Hand Tracking
Real-time detection and tracking of 21 hand landmarks using MediaPipe's ML models

### 🌀 Portal Rendering
Dynamic magical circles with smooth rotation and scaling animations

### ✨ Particle System
Mystical sparks emanating from portals and fingertips

</td>
<td width="50%">

### 🎨 Energy Trails
Motion-based trails following hand movements

### 🔮 Runic Symbols
Ancient symbols orbiting each active portal

### ⚡ Energy Beams
Connecting beam effects between dual portals

</td>
</tr>
</table>

---

## 🛠️ Technical Stack

```
MediaPipe → Hand Landmark Detection (21 points/hand)
    ↓
Palm Openness Calculation (finger-to-wrist distances)
    ↓
Portal Activation & Rendering (OpenCV affine transformations)
    ↓
Real-time Visual Effects (particles, trails, beams)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rudra-Tiwari-codes/Doctor-Strange.git
cd Doctor-Strange

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python main.py
```

---

## 🎮 Controls

| Action | Effect |
|--------|--------|
| 🖐️ **Open Palm** | Activate magical portal |
| ✌️ **Two Palms** | Generate energy beam connection |
| ⌨️ **Press 'q'** | Exit application |

---

## ⚙️ Configuration

Customize the experience by editing `config.json`:

```json
{
  "camera": { "width": 1280, "height": 720 },
  "line_settings": { "color": [0, 140, 255] },
  "overlay": { "rotation_degree_increment": 5 }
}
```

---

## 📋 Requirements

- **Python** 3.8 - 3.12
- **OpenCV** 4.x
- **MediaPipe** 0.10+
- **NumPy** 1.x

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with 🔮 by Rudra Tiwari**

*May the Mystic Arts be with you*

</div>
