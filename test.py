import cv2
import face_recognition
import numpy as np
import time

cap = cv2.VideoCapture(0)
print("Warming up the webcam...")

# Initial delay to allow camera to adjust
time.sleep(2)

frame_count = 0
print("Testing webcam input for face_recognition...")

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret or frame is None:
        print("Frame not received.")
        continue

    # Add a small delay every frame
    time.sleep(0.1)

    if frame.dtype != np.uint8:
        print("Wrong dtype:", frame.dtype)
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if small_frame.shape[0] == 0 or small_frame.shape[1] == 0:
        print("Empty frame after resize")
        continue

    try:
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("Error converting to RGB:", e)
        continue

    try:
        face_locations = face_recognition.face_locations(rgb_small)
        print(f"[{frame_count}] {len(face_locations)} face(s) detected.")
    except Exception as e:
        print(f"[{frame_count}] face_recognition failed:", e)
        continue

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
