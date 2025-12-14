"""
Utility Functions for AI Sign Language to Speech Project

This file contains helper functions used across multiple modules.
These are common operations that we don't want to repeat in every file.

BEGINNER NOTE: These are like tools in a toolbox - other files will import
and use these functions when needed.
"""

import cv2
import numpy as np
import os
from datetime import datetime


def normalize_landmarks(hand_landmarks):
    """
    Convert MediaPipe hand landmarks to a normalized numpy array.
    
    MediaPipe gives us 21 landmarks (points) on the hand, each with x, y, z coordinates.
    We normalize them so they're relative to the wrist (landmark 0).
    
    WHY: This makes the model work regardless of hand size or distance from camera.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        numpy array of shape (63,) containing normalized x, y, z for all 21 landmarks
        Returns None if landmarks are invalid
    """
    if hand_landmarks is None:
        return None
    
    # Extract all landmark coordinates into a list
    # Each landmark has x, y, z (3 values) × 21 landmarks = 63 values total
    landmark_list = []
    
    for landmark in hand_landmarks.landmark:
        landmark_list.append(landmark.x)
        landmark_list.append(landmark.y)
        landmark_list.append(landmark.z)
    
    # Convert to numpy array for easier math operations
    landmark_array = np.array(landmark_list)
    
    # Normalize: subtract the wrist position (first landmark) from all landmarks
    # This makes coordinates relative to the wrist, not the screen
    wrist_x = landmark_array[0]
    wrist_y = landmark_array[1]
    wrist_z = landmark_array[2]
    
    # Subtract wrist coordinates from all x, y, z values
    for i in range(0, len(landmark_array), 3):
        landmark_array[i] -= wrist_x      # Normalize x
        landmark_array[i + 1] -= wrist_y  # Normalize y
        landmark_array[i + 2] -= wrist_z  # Normalize z
    
    return landmark_array


def draw_landmarks_on_image(image, hand_landmarks, mp_hands, mp_drawing):
    """
    Draw hand landmarks and connections on the image.
    
    This makes it easy to see if hand detection is working correctly.
    
    Args:
        image: The camera frame (numpy array)
        hand_landmarks: MediaPipe hand landmarks object
        mp_hands: MediaPipe hands module
        mp_drawing: MediaPipe drawing utilities module
        
    Returns:
        Image with landmarks drawn on it
    """
    if hand_landmarks is None:
        return image
    
    # Make a copy so we don't modify the original
    annotated_image = image.copy()
    
    # Draw the landmarks (dots) and connections (lines)
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    
    return annotated_image


def put_text_on_image(image, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    """
    Put text on an image with a black background for better readability.
    
    Args:
        image: The image to draw on
        text: The text to display
        position: Tuple (x, y) for text position
        color: Text color in BGR format
        font_scale: Size of the text
        thickness: Thickness of the text
        
    Returns:
        Image with text drawn on it
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size to draw background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw black rectangle as background
    cv2.rectangle(
        image,
        (position[0], position[1] - text_height - baseline),
        (position[0] + text_width, position[1] + baseline),
        (0, 0, 0),
        -1  # Filled rectangle
    )
    
    # Draw text on top of rectangle
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    return image


def create_csv_header():
    """
    Create the header row for CSV files.
    
    We have 63 features (21 landmarks × 3 coordinates) plus 1 label column.
    
    Returns:
        List of column names for the CSV
    """
    header = []
    
    # Create column names: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
    for i in range(21):
        header.append(f'x{i}')
        header.append(f'y{i}')
        header.append(f'z{i}')
    
    # Add label column at the end
    header.append('label')
    
    return header


def save_landmark_to_csv(landmark_array, label, csv_path):
    """
    Save a single landmark sample to a CSV file.
    
    Args:
        landmark_array: Normalized landmark array (63 values)
        label: The letter this gesture represents (A-Z)
        csv_path: Path to the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if file exists
        file_exists = os.path.isfile(csv_path)
        
        # Open file in append mode
        with open(csv_path, 'a') as f:
            # Write header if file is new
            if not file_exists:
                header = create_csv_header()
                f.write(','.join(header) + '\n')
            
            # Convert landmark array to string and add label
            landmark_str = ','.join(map(str, landmark_array))
            row = f"{landmark_str},{label}\n"
            
            # Write the row
            f.write(row)
        
        return True
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def load_all_csv_data(data_dir):
    """
    Load all CSV files from the data directory.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Tuple of (features, labels) as numpy arrays
        Returns (None, None) if no data found
    """
    import pandas as pd
    
    all_data = []
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None
    
    # Load each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        return None, None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Separate features (landmarks) and labels
    features = combined_df.iloc[:, :-1].values  # All columns except last
    labels = combined_df.iloc[:, -1].values     # Last column
    
    return features, labels


def get_timestamp():
    """
    Get current timestamp as a formatted string.
    
    Returns:
        String like "2024-01-15_14-30-45"
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def calculate_fps(prev_time):
    """
    Calculate frames per second.
    
    Args:
        prev_time: Previous frame time from time.time()
        
    Returns:
        Tuple of (fps, current_time)
    """
    import time
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    return fps, current_time


def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    """
    Print a progress bar to the console.
    
    Args:
        iteration: Current iteration (0 to total)
        total: Total iterations
        prefix: Text before the bar
        suffix: Text after the bar
        length: Length of the bar in characters
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()  # New line when complete
