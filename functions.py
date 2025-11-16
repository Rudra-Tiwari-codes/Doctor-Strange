"""Computer Vision Functions for Doctor Strange AR Filter

Core image processing and overlay functions for hand tracking visualization.
Author: Rudra Tiwari
Date: November 2025
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple
import random
import math

LINE_COLOR = (0, 140, 255)
WHITE_COLOR = (255, 255, 255)
ORANGE_GLOW = (0, 140, 255)  # BGR format


class Particle:
    """Represents a single magical particle/spark."""
    
    def __init__(self, x: int, y: int, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = 1.0
        self.size = random.randint(2, 6)
        self.color = (
            random.randint(0, 100),
            random.randint(100, 200),
            random.randint(200, 255)
        )
    
    def update(self):
        """Update particle position and life."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.02
        self.vy += 0.3  # Gravity effect
        return self.life > 0
    
    def draw(self, frame: np.ndarray):
        """Draw the particle on the frame."""
        if self.life > 0:
            alpha = int(self.life * 255)
            size = int(self.size * self.life)
            cv.circle(frame, (int(self.x), int(self.y)), size, self.color, -1)


class ParticleSystem:
    """Manages multiple particles for magical effects."""
    
    def __init__(self):
        self.particles = []
    
    def emit(self, x: int, y: int, count: int = 5):
        """Emit particles from a specific position."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(Particle(x, y, vx, vy))
    
    def update(self, frame: np.ndarray):
        """Update and draw all particles."""
        self.particles = [p for p in self.particles if p.update()]
        for particle in self.particles:
            particle.draw(frame)


class EnergyTrail:
    """Creates energy trails that follow hand movement."""
    
    def __init__(self, max_points: int = 20):
        self.trail_points = []
        self.max_points = max_points
    
    def add_point(self, point: Tuple[int, int]):
        """Add a new point to the trail."""
        self.trail_points.append(point)
        if len(self.trail_points) > self.max_points:
            self.trail_points.pop(0)
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 200, 255)):
        """Draw the energy trail with fading effect."""
        if len(self.trail_points) < 2:
            return
        
        for i in range(1, len(self.trail_points)):
            # Calculate alpha based on position in trail
            alpha = i / len(self.trail_points)
            thickness = int(3 + alpha * 5)
            
            # Draw line segment with glow
            cv.line(frame, self.trail_points[i-1], self.trail_points[i], 
                   color, thickness, cv.LINE_AA)
            # Inner bright line
            cv.line(frame, self.trail_points[i-1], self.trail_points[i], 
                   (255, 255, 255), max(1, thickness // 2), cv.LINE_AA)
    
    def clear(self):
        """Clear the trail."""
        self.trail_points = []


class RunicSymbol:
    """Animated runic symbols that orbit around portals."""
    
    def __init__(self, center_x: int, center_y: int, radius: int, angle_offset: float):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.angle = angle_offset
        self.rotation_speed = 0.05
        self.symbol_char = random.choice(['ᚠ', 'ᚢ', 'ᚦ', 'ᚨ', 'ᚱ', 'ᚲ', 'ᚷ', 'ᚹ'])
    
    def update(self):
        """Update symbol position."""
        self.angle += self.rotation_speed
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 180, 255)):
        """Draw the runic symbol."""
        x = int(self.center_x + self.radius * math.cos(self.angle))
        y = int(self.center_y + self.radius * math.sin(self.angle))
        
        # Draw symbol with glow effect
        cv.putText(frame, self.symbol_char, (x-10, y+10), 
                  cv.FONT_HERSHEY_COMPLEX, 0.8, color, 2, cv.LINE_AA)
        cv.putText(frame, self.symbol_char, (x-10, y+10), 
                  cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)


def apply_mirror_dimension_effect(frame: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    Apply a reality-bending mirror dimension effect.
    
    Args:
        frame: Input frame
        intensity: Effect intensity (0.0 to 1.0)
    
    Returns:
        Frame with mirror dimension effect applied
    """
    h, w = frame.shape[:2]
    
    # Create a subtle distortion effect
    # Split frame into sections and apply geometric transformations
    center_x, center_y = w // 2, h // 2
    
    # Create kaleidoscope effect by mirroring quadrants with reduced opacity
    overlay = frame.copy()
    
    # Create distortion maps for subtle warping
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            dx = j - center_x
            dy = i - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Subtle spiral distortion
                angle = np.arctan2(dy, dx)
                angle += distance * 0.00015 * intensity
                map_x[i, j] = center_x + distance * np.cos(angle)
                map_y[i, j] = center_y + distance * np.sin(angle)
            else:
                map_x[i, j] = j
                map_y[i, j] = i
    
    # Apply distortion
    warped = cv.remap(frame, map_x, map_y, cv.INTER_LINEAR)
    
    # Blend with original for subtle effect
    cv.addWeighted(warped, intensity * 0.5, frame, 1 - intensity * 0.5, 0, frame)
    
    return frame

def position_data(lmlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Extracts and returns key fingertip and hand positions from the landmark list.

    Args:
        lmlist (list): List of tuples containing (x, y) coordinates of keypoints.

    Returns:
        list: Coordinates of wrist, thumb tip, index mcp, index tip,
              middle mcp, middle tip, ring tip, and pinky tip.
    """
    if len(lmlist) < 21:
        raise ValueError("Landmark list must contain at least 21 points.")

    keys = [0, 4, 5, 8, 9, 12, 16, 20]  # Indices of relevant landmarks
    return [lmlist[i] for i in keys]

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (tuple): First point (x1, y1).
        p2 (tuple): Second point (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    # Use numpy for faster computation
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int] = LINE_COLOR,
    thickness: int = 5
) -> np.ndarray:
    """
    Draws a line between two points on the given frame.

    Args:
        frame (ndarray): The image/frame to draw the line on.
        p1 (tuple): First point (x1, y1).
        p2 (tuple): Second point (x2, y2).
        color (tuple): Color of the outer line.
        thickness (int): Thickness of the outer line.

    Returns:
        ndarray: Modified frame with the drawn line.
    """
    cv.line(frame, p1, p2, color, thickness)
    cv.line(frame, p1, p2, WHITE_COLOR, max(1, thickness // 2))
    return frame


def add_glow_effect(frame: np.ndarray, x: int, y: int, radius: int, color: Tuple[int, int, int], intensity: float = 0.3) -> np.ndarray:
    """
    Add a subtle glowing aura effect around a point (Marvel-style).
    
    Args:
        frame: The image/frame to add glow to
        x: X-coordinate of glow center
        y: Y-coordinate of glow center
        radius: Radius of the glow effect
        color: BGR color tuple for the glow
        intensity: Glow intensity (0.0 to 1.0)
    
    Returns:
        Modified frame with glow effect
    """
    overlay = frame.copy()
    
    # Create subtle multi-layer glow for professional look
    for i in range(3, 0, -1):
        alpha = intensity * 0.15 * i
        glow_radius = int(radius * (1 + i * 0.08))
        cv.circle(overlay, (x, y), glow_radius, color, -1)
    
    # Blend with original frame for subtle effect
    cv.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


def detect_gesture(lmlist: List[Tuple[int, int]]) -> str:
    """
    Detect specific hand gestures for different magical effects.
    
    Args:
        lmlist: List of hand landmark coordinates
    
    Returns:
        String representing detected gesture ('fist', 'peace', 'open', 'point', 'none')
    """
    if len(lmlist) < 21:
        return 'none'
    
    # Calculate finger states (extended or closed)
    fingers_up = []
    
    # Thumb
    if lmlist[4][0] > lmlist[3][0]:  # Right hand
        fingers_up.append(lmlist[4][0] > lmlist[2][0])
    else:  # Left hand
        fingers_up.append(lmlist[4][0] < lmlist[2][0])
    
    # Other fingers
    for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_up.append(lmlist[finger_tip][1] < lmlist[finger_pip][1])
    
    # Gesture detection
    if sum(fingers_up) == 0:
        return 'fist'
    elif fingers_up == [False, True, True, False, False]:
        return 'peace'
    elif sum(fingers_up) >= 4:
        return 'open'
    elif fingers_up == [False, True, False, False, False]:
        return 'point'
    
    return 'partial'


def overlay_image(
    target_img: np.ndarray,
    frame: np.ndarray,
    x: int, y: int,
    size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Overlays a target image onto a frame at the given position.

    Args:
        target_img (ndarray): Image to be overlaid with an alpha channel.
        frame (ndarray): Frame on which the target image is to be overlaid.
        x (int): X-coordinate for the top-left corner of the overlay.
        y (int): Y-coordinate for the top-left corner of the overlay.
        size (tuple, optional): Size to resize the target image. Defaults to None.

    Returns:
        ndarray: Modified frame with the overlaid image.
    """
    if size:
        try:
            target_img = cv.resize(target_img, size)
        except cv.error as e:
            raise ValueError(f"Error resizing the target image: {e}")

    if target_img.shape[-1] != 4:
        raise ValueError("Target image must have 4 channels (RGBA).")

    # Split the RGBA channels
    b, g, r, a = cv.split(target_img)
    overlay_color = cv.merge((b, g, r))
    mask = cv.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    
    # Ensure coordinates are valid
    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame  # Skip overlay if out of bounds

    roi = frame[y:y + h, x:x + w]
    
    # Ensure mask dimensions match ROI
    if roi.shape[0] != mask.shape[0] or roi.shape[1] != mask.shape[1]:
        return frame  # Skip if dimensions don't match

    # Create background and foreground masks
    img1_bg = cv.bitwise_and(roi, roi, mask=cv.bitwise_not(mask))
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Overlay the foreground onto the background
    frame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    return frame
