# Doctor Strange AR Filter

![Python](https://img.shields.io/badge/Python-3.8--3.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A6D6)

## Problem Statement

AR filters on social platforms are locked to their ecosystems and require app downloads. Developers and content creators need a standalone, customizable AR solution that runs on any webcam without cloud dependencies.

## Solution

A real-time augmented reality filter that renders Doctor Strange-style magic portals and energy effects following hand gestures, running entirely on local hardware.

## Methodology

- **Hand Detection** — MediaPipe extracts 21 hand landmarks per frame
- **Gesture Recognition** — Custom classifier identifies spell-casting poses
- **Effect Rendering** — OpenCV overlays particle systems, energy trails, and portal animations
- **Performance Optimization** — Frame pipeline optimized for 30+ FPS on CPU

## Results

| Metric | Value |
|--------|-------|
| Frame Rate | 30+ FPS |
| Latency | <50ms |
| Hand Tracking | 21 landmarks |
| Effects | Portals, particles, energy trails |

## Usage

```bash
pip install -r requirements.txt
python main.py
```

| Key | Action |
|-----|--------|
| `r` | Start recording |
| `q` | Quit |

## Demo

[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/YOUR_VIDEO_ID)

## Future Improvements

- Add two-hand portal interactions for portal-to-portal transport effects
- Implement face tracking for mask-based AR effects

---

**Made by [Rudra Tiwari](https://github.com/Rudra-Tiwari-codes)**
