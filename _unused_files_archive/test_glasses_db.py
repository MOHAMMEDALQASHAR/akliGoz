import sqlite3
import os
import pickle

db_path = "web_face_recognition/instance/face_recognition.db"

print("="*60)
print("CHECKING DATABASE CONNECTION FOR GLASSES")
print("="*60)
print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face'")
    if cursor.fetchone():
        print("[OK] 'face' table found")
        
        # Get all faces
        cursor.execute("SELECT id, name, image_path, encoding FROM face")
        rows = cursor.fetchall()
        
        print(f"\nFaces in database: {len(rows)}")
        for row in rows:
            face_id, name, img_path, encoding_blob = row
            print(f"\n  Face ID: {face_id}")
            print(f"  Name: {name}")
            print(f"  Image: {img_path}")
            print(f"  Image exists: {os.path.exists(img_path)}")
            
            if encoding_blob:
                try:
                    enc = pickle.loads(encoding_blob)
                    print(f"  Encoding shape: {enc.shape if hasattr(enc, 'shape') else len(enc)}")
                    print(f"  Encoding type: {type(enc)}")
                except Exception as e:
                    print(f"  ERROR loading encoding: {e}")
            else:
                print(f"  WARNING: No encoding saved!")
    else:
        print("[ERROR] 'face' table NOT found")
    
    conn.close()
else:
    print("[ERROR] Database file does NOT exist")

print("\n" + "="*60)
