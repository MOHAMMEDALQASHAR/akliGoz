import cv2
import numpy as np
import sqlite3
import pickle
import os

print("="*60)
print("FACE RECOGNITION TEST")
print("="*60)

# Initialize face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load from database
db_path = "web_face_recognition/instance/face_recognition.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, name, image_path, encoding FROM face")
faces = cursor.fetchall()

print(f"\nLoaded {len(faces)} face(s) from database")

known_names = []
known_encodings = []

for face_id, name, img_path, encoding_blob in faces:
    print(f"\n  Face: {name}")
    print(f"    Image: {img_path}")
    
    if encoding_blob:
        enc = pickle.loads(encoding_blob)
        known_encodings.append(enc)
        known_names.append(name)
        print(f"    Encoding loaded: {len(enc)} values")

conn.close()

if len(known_encodings) == 0:
    print("\nERROR: No encodings found in database!")
    exit(1)

print(f"\n{len(known_encodings)} encoding(s) ready for comparison")

# Initialize camera
print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print("\nCamera ready! Looking for faces...")
print("Press 'q' to quit\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Only process every 5th frame for performance
    if frame_count % 5 != 0:
        cv2.imshow('Face Recognition Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Convert to grayscale and apply histogram equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces_detected:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (200, 200))
        current_encoding = face_roi_resized.flatten()
        
        # Compare with known faces
        best_match = "Unknown"
        best_score = 0
        
        for i, (known_enc, known_name) in enumerate(zip(known_encodings, known_names)):
            try:
                # Normalize
                f1_norm = (current_encoding - np.mean(current_encoding)) / (np.std(current_encoding) + 1e-10)
                f2_norm = (known_enc - np.mean(known_enc)) / (np.std(known_enc) + 1e-10)
                
                # Calculate correlation
                correlation = np.corrcoef(f1_norm, f2_norm)[0, 1]
                
                print(f"  Comparing with {known_name}: {correlation:.3f}")
                
                if correlation > best_score:
                    best_score = correlation
                    if correlation > 0.60:  # Threshold
                        best_match = known_name
            except Exception as e:
                print(f"  Error comparing with {known_name}: {e}")
        
        # Draw box and name
        color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        text = f"{best_match} ({best_score:.2f})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if best_match != "Unknown":
            print(f"\n*** MATCH FOUND: {best_match} (score: {best_score:.3f}) ***\n")
    
    cv2.imshow('Face Recognition Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nTest completed")
