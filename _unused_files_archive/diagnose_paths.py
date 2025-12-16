
import sqlite3
import os

# Paths based on my understanding of app.py configuration
basedir = os.path.abspath(os.getcwd())
db_path = os.path.join(basedir, 'faces.db')
upload_dir = os.path.join(basedir, 'web_face_recognition', 'static', 'uploads', 'faces')

print("--- DEBUG DIAGNOSTIC ---")
print(f"Checking DB at: {db_path}")
if os.path.exists(db_path):
    print("YES DB file exists.")
else:
    print("NO DB file MISSING.")

print(f"Checking Upload Dir at: {upload_dir}")
if os.path.exists(upload_dir):
    print("YES Upload dir exists.")
    files = os.listdir(upload_dir)
    print(f"   Files found ({len(files)}): {files}")
else:
    print("NO Upload dir MISSING.")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, image_path FROM face")
        rows = cursor.fetchall()
        print("\n--- DB Records ---")
        for row in rows:
            print(f"ID: {row[0]}, Name: {row[1]}")
            print(f"   Stored Path: {row[2]}")
            
            # Check if this file exists
            if os.path.exists(row[2]):
                print("   YES File exists on disk.")
            else:
                print("   NO File NOT found on disk.")
                
            # Check if filename matches what we see in upload dir
            filename = os.path.basename(row[2])
            if os.path.exists(os.path.join(upload_dir, filename)):
                print(f"   YES File found in upload_dir: {filename}")
            else:
                print(f"   NO File NOT found in upload_dir: {filename}")
        
        if not rows:
            print("No records found in DB.")
            
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")
