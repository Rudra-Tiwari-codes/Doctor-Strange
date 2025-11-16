# Doctor Strange AR Filter

Real-time AR filter that renders magical portals on open palms using computer vision.

<p align="center">
  <img src="assets/doctor_strange_1.jpg" width="45%" alt="Doctor Strange Portal 1" />
  <img src="assets/doctor_strange_2.jpg" width="45%" alt="Doctor Strange Portal 2" />
</p>

## Features

- Hand tracking with MediaPipe
- Rotating portal overlays on open palms
- Particle effects and energy trails
- Runic symbols orbiting portals
- Energy beams connecting dual portals
- Sound effects support (optional)

## Technical Stack

MediaPipe detects 21 hand landmarks per hand. Palm openness is calculated using finger-to-wrist distances. Portals render when the palm opens fully, with rotation applied via affine transformations.

## Installation

```bash
git clone https://github.com/Rudra-Tiwari-codes/Doctor-Strange.git
cd Doctor-Strange
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

**Controls:**
- Open palm → Activate portal
- Two palms → Energy beam
- Press 'q' → Quit

## Configuration

Edit `config.json` for camera settings, colors, and rotation speed.

## Requirements

Python 3.8-3.12, OpenCV, MediaPipe, NumPy

## License

MIT License
