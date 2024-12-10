import cv2
import mediapipe as mp
import serial
import time
import threading
import os
from flask import Flask, jsonify
from flask_cors import CORS

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Check if the application is running in a cloud environment
IS_CLOUD = os.getenv("IS_CLOUD", "false").lower() == "true"

# Initialize Serial Communication only if not running in a cloud environment
ser = None
if not IS_CLOUD:
    try:
        ser = serial.Serial('COM8', 9600, timeout=1)  # COM6 for local serial communication
        print("Serial port initialized.")
    except serial.SerialException:
        print("Warning: Serial port COM6 not found. Servo control will be disabled.")
else:
    print("Running in cloud environment. Serial port is disabled.")

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to track face detection status
face_detected = False
last_face_detection = False
servo_moved = False  # New flag to track servo movement

# Function to move the servo
def move_servo():
    """Non-blocking function to move the servo via serial communication."""
    if ser and ser.isOpen():  # Ensure the serial port is open
        threading.Thread(target=_servo_control, daemon=True).start()

def _servo_control():
    if ser:
        print("Moving servo to active position...")
        ser.write(b'f')  # Command to move servo to active position
        time.sleep(2)    # Wait for 1 second
        print("Returning servo to home position...")
        ser.write(b'h')  # Command to return servo to home position
    else:
        print("Servo control is disabled as no serial port is available.")

# Flask route to get face detection status
@app.route('/detect_face')
def detect_face():
    global last_face_detection
    if face_detected and not last_face_detection:
        last_face_detection = True
        return jsonify({'face_detected': True})
    elif not face_detected and last_face_detection:
        last_face_detection = False
        return jsonify({'face_detected': False})
    return jsonify({'face_detected': face_detected})

# Function to run face detection
def run_face_detection():
    global face_detected, servo_moved
    cap = None

    # Only open the camera if running locally
    if not IS_CLOUD:
        cap = cv2.VideoCapture(0)  # 0 for webcam

        # Ensure the camera is accessible
        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            if not IS_CLOUD:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to read from camera.")
                    break

                # Convert the image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Detect faces
                results = face_detection.process(image)

                # Check if any face is detected
                face_detected = bool(results.detections)

                # Move the servo only when a face is detected for the first time
                if face_detected:
                    if not servo_moved:  # Ensure the servo moves only once
                        move_servo()
                        servo_moved = True  # Set the flag to prevent repeated movements
                else:
                    servo_moved = False  # Reset the flag when no face is detected
            else:
                # If in the cloud, just simulate face detection (no camera)
                face_detected = False

            # Exit on ESC key if running locally
            if not IS_CLOUD and cv2.waitKey(5) & 0xFF == 27:
                break

    if cap:
        cap.release()

# Run the face detection function immediately when the Flask app starts
def start_face_detection():
    threading.Thread(target=run_face_detection, daemon=True).start()

# Run Flask app in a separate thread to avoid blocking the main thread
def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

# Main entry point
if __name__ == "__main__":
    start_face_detection()  # Start face detection
    run_flask_app()         # Start Flask app
