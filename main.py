import os
import cv2
import mediapipe as mp
import numpy as np
import math
import winsound
import logging

# Define the log directory
log_directory = 'logs'  # You can change this to your desired path

# Create the log directory if it doesn't exist
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Set up logging configuration
log_file_path = os.path.join(log_directory, 'activity_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Take live camera input for pose detection
cap = cv2.VideoCapture(0)

# Set the desired frame rate (e.g., 10 frames per second)
frame_rate = 10

# Thresholds for activity detection (adjust as needed)
crouch_threshold = 0.1
choking_threshold = 0.2

# Number of frames to use for smoothing knee movement
smooth_window = 5
left_knee_buffer = [0] * smooth_window
right_knee_buffer = [0] * smooth_window

def detect_activity(results):
    """Detects activities based on pose landmarks."""
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    hip_knee_distance = abs(left_hip.y - left_knee.y) + abs(right_hip.y - right_knee.y)

    # Detect crawling activity
    if left_elbow.y > left_knee.y or right_elbow.y > right_knee.y:
        logging.info("Crawling detected")
        print("CRAWLING")
        winsound.Beep(1000, 200)

    # Detect crouching activity
    if hip_knee_distance < crouch_threshold:
        logging.info("Crouch position detected")
        print("Crouch position detected")
        winsound.Beep(1000, 200)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (600, 400))
    
    # Process the image for pose detection
    results = pose.process(img)

    if results.pose_landmarks:
        detect_activity(results)  # Call the activity detection function

        # Draw landmarks on the image
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

    cv2.imshow("Pose Estimation", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()