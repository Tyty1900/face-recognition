import cv2
import face_recognition
import numpy as np

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Training... Look at the camera.")

for i in range(5):
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print(f"Error: Couldn't read frame {i+1}")
        continue

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check if the image format is correct
    print(f"Frame {i+1} shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")

    # Use OpenCV's face detector to confirm face detection works
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    if len(faces) > 0:
        print(f"OpenCV detected {len(faces)} face(s) in frame {i+1}")
    else:
        print(f"No face detected in frame {i+1} using OpenCV")

    # Use face_recognition to detect face locations
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            print(f"Face_recognition detected {len(face_locations)} face(s) in frame {i+1}")
        else:
            print(f"No face detected in frame {i+1} using face_recognition")
    except Exception as e:
        print(f"Error using face_recognition in frame {i+1}: {e}")

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
