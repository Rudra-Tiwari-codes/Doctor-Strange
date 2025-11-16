"""Doctor Strange AR Filter - Main Application

Real-time hand tracking and magical circle overlay application.
Author: Rudra Tiwari
Date: November 2025
"""

import cv2 as cv
import mediapipe as mp
import json
from functions import (position_data, calculate_distance, draw_line, overlay_image, 
                       ParticleSystem, add_glow_effect, detect_gesture)
import numpy as np

def load_config(path: str = "config.json") -> dict:
    """Loads the configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing application configuration
    """
    with open(path, "r") as file:
        return json.load(file)

def limit_value(val: int, min_val: int, max_val: int) -> int:
    """Clamps a value between a given min and max."""
    return max(min(val, max_val), min_val)

def initialize_camera(config: dict) -> cv.VideoCapture:
    """Initializes the webcam with given width, height, and device ID."""
    cap = cv.VideoCapture(config["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    if not cap.isOpened():
        raise RuntimeError("Failed to open the webcam.")
    return cap

def load_images(config: dict) -> tuple:
    """Loads overlay images and raises an error if any are missing."""
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("Failed to load one or more overlay images.")
    return inner_circle, outer_circle

def process_frame(frame, hands, config, inner_circle, outer_circle, deg, particle_system, portal_scales):
    """Processes the frame, applies overlays, and returns the updated frame."""
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            (wrist, thumb_tip, index_mcp, index_tip,
             middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinky_distance = calculate_distance(index_tip, pinky_tip)
            ratio = index_pinky_distance / index_wrist_distance
            
            # Detect gesture for special effects
            gesture = detect_gesture(lm_list)

            if 0.5 < ratio < 1.3:
                fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                
                # Enhanced line drawing with glow based on gesture
                line_color = tuple(config["line_settings"]["color"])
                if gesture == 'peace':
                    line_color = (255, 100, 0)  # Blue glow for peace sign
                elif gesture == 'fist':
                    line_color = (0, 0, 255)  # Red glow for fist
                
                for finger in fingers:
                    frame = draw_line(frame, wrist, finger,
                                      color=line_color,
                                      thickness=config["line_settings"]["thickness"])
                for i in range(len(fingers) - 1):
                    frame = draw_line(frame, fingers[i], fingers[i + 1],
                                      color=line_color,
                                      thickness=config["line_settings"]["thickness"])
                
                # Emit particles at fingertips for dramatic effect
                if gesture == 'peace':
                    particle_system.emit(index_tip[0], index_tip[1], 3)
                    particle_system.emit(middle_tip[0], middle_tip[1], 3)

            elif ratio >= 1.3:
                center_x, center_y = middle_mcp
                diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])

                x1 = limit_value(center_x - diameter // 2, 0, w)
                y1 = limit_value(center_y - diameter // 2, 0, h)
                diameter = min(diameter, w - x1, h - y1)
                
                # Portal opening animation - gradually scale up
                target_scale = 1.0
                if hand_idx not in portal_scales:
                    portal_scales[hand_idx] = 0.3
                
                portal_scales[hand_idx] = min(portal_scales[hand_idx] + 0.05, target_scale)
                current_diameter = int(diameter * portal_scales[hand_idx])
                current_x1 = center_x - current_diameter // 2
                current_y1 = center_y - current_diameter // 2

                deg = (deg + config["overlay"]["rotation_degree_increment"]) % 360
                M1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), deg, 1.0)
                M2 = cv.getRotationMatrix2D((inner_circle.shape[1] // 2, inner_circle.shape[0] // 2), -deg, 1.0)

                rotated_outer = cv.warpAffine(outer_circle, M1, (outer_circle.shape[1], outer_circle.shape[0]))
                rotated_inner = cv.warpAffine(inner_circle, M2, (inner_circle.shape[1], inner_circle.shape[0]))
                
                # Add glow effect around portal
                frame = add_glow_effect(frame, center_x, center_y, current_diameter // 2, (0, 165, 255))

                frame = overlay_image(rotated_outer, frame, current_x1, current_y1, (current_diameter, current_diameter))
                frame = overlay_image(rotated_inner, frame, current_x1, current_y1, (current_diameter, current_diameter))
                
                # Emit particles around the portal edge
                for angle in range(0, 360, 30):
                    rad = np.radians(angle)
                    px = int(center_x + (current_diameter // 2) * np.cos(rad))
                    py = int(center_y + (current_diameter // 2) * np.sin(rad))
                    particle_system.emit(px, py, 2)
    
    # Update and draw all particles
    particle_system.update(frame)

    return frame, deg

def main():
    """Main function to run the application."""
    config = load_config()
    cap = initialize_camera(config)
    inner_circle, outer_circle = load_images(config)
    hands = mp.solutions.hands.Hands()
    deg = 0
    
    # Initialize particle system and portal animation tracking
    particle_system = ParticleSystem()
    portal_scales = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            frame = cv.flip(frame, 1)
            frame, deg = process_frame(frame, hands, config, inner_circle, outer_circle, deg, particle_system, portal_scales)

            cv.imshow("Image", frame)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
