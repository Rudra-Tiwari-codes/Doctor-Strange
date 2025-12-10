"""Computer Vision Functions for Doctor Strange AR Filter

Core image processing and overlay functions for hand tracking visualization.
Author: Rudra Tiwari
Date: November 2025
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple

try:
    import pygame
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

LINE_COLOR = (0, 140, 255)
WHITE_COLOR = (255, 255, 255)
ORANGE_GLOW = (0, 140, 255)  # BGR format


class SoundManager:
    """Manages sound effects for the application."""
    
    def __init__(self):
        self.enabled = SOUND_AVAILABLE
        self.music_playing = False
        if self.enabled:
            try:
                pygame.mixer.init()
                self.portal_open_sound = None
                self.gesture_sound = None
                self.throw_sound = None
            except (pygame.error, OSError):
                self.enabled = False
    
    def load_sounds(self):
        """Load sound files if they exist."""
        if not self.enabled:
            return
        
        try:
            # Try to load sounds if they exist in sounds folder
            # Users can add: portal_open.wav, gesture.wav, throw.wav, doctor_strange_theme.mp3
            import os
            if os.path.exists("sounds/doctor_strange_theme.mp3"):
                pygame.mixer.music.load("sounds/doctor_strange_theme.mp3")
                pygame.mixer.music.set_volume(0.3)
        except (pygame.error, OSError, FileNotFoundError):
            pass
    
    def play_music(self):
        """Play background music on loop."""
        if self.enabled and not self.music_playing:
            try:
                pygame.mixer.music.play(-1)  # Loop indefinitely
                self.music_playing = True
            except pygame.error:
                pass
    
    def stop_music(self):
        """Stop background music."""
        if self.enabled and self.music_playing:
            try:
                pygame.mixer.music.stop()
                self.music_playing = False
            except pygame.error:
                pass
    
    def play_portal_open(self):
        """Play portal opening sound."""
        if self.enabled and self.portal_open_sound:
            try:
                self.portal_open_sound.play()
            except pygame.error:
                pass
    
    def play_throw(self):
        """Play disc throw sound effect."""
        if self.enabled and self.throw_sound:
            try:
                self.throw_sound.play()
            except pygame.error:
                pass
    
    def play_gesture(self):
        """Play gesture sound effect."""
        if self.enabled and self.gesture_sound:
            try:
                self.gesture_sound.play()
            except pygame.error:
                pass


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




def draw_energy_beam(frame: np.ndarray, point1: Tuple[int, int], point2: Tuple[int, int], 
                     color: Tuple[int, int, int] = (0, 200, 255)) -> np.ndarray:
    """
    Draw an energy beam connecting two portals.
    
    Args:
        frame: Frame to draw on
        point1: First portal center
        point2: Second portal center
        color: Beam color
    
    Returns:
        Frame with energy beam
    """
    # Draw thick outer glow
    cv.line(frame, point1, point2, color, 12, cv.LINE_AA)
    # Draw medium inner line
    cv.line(frame, point1, point2, (100, 220, 255), 6, cv.LINE_AA)
    # Draw bright core
    cv.line(frame, point1, point2, (255, 255, 255), 2, cv.LINE_AA)
    
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
