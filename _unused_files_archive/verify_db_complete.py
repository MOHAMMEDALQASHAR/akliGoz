import sqlite3
import pickle
import numpy as np
import os

db_path = "web_face_recognition/instance/face_recognition.db"

print("="*60)
print("COMPLETE DATABASE VERIFICATION")
print("="*60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, name, image_path, encoding FROM face")
faces = cursor.fetchall()

print(f"\nTotal faces: {len(faces)}")

for face_id, name, img_path, encoding_blob in faces:
    print(f"\n--- FACE ID: {face_id} ---")
    print(f"Name: {name}")
    print(f"Image: {img_path}")
    print(f"Image exists: {os.path.exists(img_path)}")
    
    if encoding_blob:
        try:
            enc = pickle.loads(encoding_blob)
            print(f"Encoding type: {type(enc)}")
            print(f"Encoding shape: {enc.shape if hasattr(enc, 'shape') else 'no shape'}")
            print(f"Encoding dtype: {enc.dtype if hasattr(enc, 'dtype') else 'no dtype'}")
            print(f"Encoding size: {enc.size if hasattr(enc, 'size') else len(enc)}")
            print(f"Encoding min/max: {enc.min():.2f} / {enc.max():.2f}")
            
            # Check if it's the right format for LBPH
            if len(enc.shape) == 2:
                print(f"[OK] 2D image format (suitable for LBPH)")
            elif len(enc.shape) == 1:
                # Try reshaping
                try:
                    reshaped = enc.reshape(200, 200)
                    print(f"[OK] Can be reshaped from 1D to 2D ({reshaped.shape})")
                except:
                    print(f"[ERROR] Cannot reshape to 200x200")
            else:
                print(f"[WARNING] Unexpected shape: {enc.shape}")
                
        except Exception as e:
            print(f"[ERROR] Cannot load encoding: {e}")
    else:
        print(f"[ERROR] No encoding blob saved!")

conn.close()

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
