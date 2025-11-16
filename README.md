# Doctor Strange AR Filter

A real-time augmented reality filter that creates magical circles on your palms using computer vision, inspired by Marvel's Doctor Strange.

## Overview

## Overview

This computer vision project leverages MediaPipe and OpenCV to detect hand landmarks from your webcam and overlay rotating magical circles on open palms. The application processes video in real-time, calculating palm position and openness to dynamically render the visual effects.

## Features

- Real-time hand tracking and landmark detection
- Dynamic palm openness calculation
- Rotating magical circle overlays
- Customizable visual effects through configuration
- Multiple color schemes available

## Technical Details

The application uses MediaPipe's hand tracking solution to detect 21 key points on each hand. By calculating distances between specific landmarks (wrist, fingertips, palm center), the system determines palm openness and position. When the palm is fully open, it renders rotating inner and outer circle overlays at the calculated center point.

## Project Structure

```
.
├── Models/
│   ├── inner_circles/     # Inner circle overlay images
│   └── outer_circles/     # Outer circle overlay images
├── functions.py           # Core image processing functions
├── main.py               # Application entry point
├── config.json           # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rudra-Tiwari-codes/Doctor-Strange.git
cd Doctor-Strange
```

2. Create a virtual environment (Python 3.8-3.12):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

- Position your hand in front of the camera
- Open your palm fully to activate the magical circles
- Press 'q' to quit

## Configuration

Edit `config.json` to customize:
- Camera resolution and device
- Line colors and thickness
- Circle overlay paths and sizes
- Rotation speed
- Keybindings

## Requirements

- Python 3.8-3.12
- OpenCV
- MediaPipe
- NumPy

## License

MIT License - See LICENSE file for details
