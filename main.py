import cv2
from deepface import DeepFace
import pandas as pd
import time
import uuid

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store face encodings and IDs
known_faces = []
face_ids = []

# To store results
log_data = []

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detect all faces and analyze emotions
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # If only one face is detected, wrap in list
        if not isinstance(results, list):
            results = [results]

        for face in results:
            # Get region of the face
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            face_img = frame[y:y+h, x:x+w]

            # Try to find if this face already exists (very basic comparison using face region similarity)
            match_found = False
            face_encoding = DeepFace.represent(face_img, enforce_detection=False)[0]['embedding']
            for i, known in enumerate(known_faces):
                dist = sum([(a - b) ** 2 for a, b in zip(face_encoding, known)]) ** 0.5
                if dist < 0.7:  # Distance threshold for face similarity
                    person_id = face_ids[i]
                    match_found = True
                    break

            if not match_found:
                person_id = f"{len(face_ids) + 1:03}"
                face_ids.append(person_id)
                known_faces.append(face_encoding)

            # Get emotion
            emotion = face['dominant_emotion']

            # Timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Add to log
            log_data.append({'Person_ID': person_id, 'Emotion': emotion, 'Timestamp': timestamp})

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {person_id} | {emotion}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detection with IDs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(log_data)
df.to_csv("emotion_log.csv", index=False)
print("Emotion data saved to emotion_log.csv")

