
import sqlite3
import os

# Define the expected path as configured in app.py
basedir = os.path.abspath(os.getcwd())
db_path = os.path.join(basedir, 'web_face_recognition', 'instance', 'face_recognition.db')

print(f"Checking database at: {db_path}")

if not os.path.exists(db_path):
    print("❌ Database file does not exist at this path!")
    # Check other locations
    alt_path = os.path.join(basedir, 'instance', 'face_recognition.db')
    print(f"Checking alternative path: {alt_path}")
    if os.path.exists(alt_path):
        print("✅ Found database at alternative path!")
        db_path = alt_path
    else:
        print("❌ Database not found anywhere standard.")
        exit()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")
    
    # Check faces
    cursor.execute("SELECT count(*) FROM face;")
    count = cursor.fetchone()[0]
    print(f"Number of faces in 'face' table: {count}")
    
    if count > 0:
        cursor.execute("SELECT id, name, image_path, created_at FROM face ORDER BY id DESC LIMIT 5;")
        rows = cursor.fetchall()
        print("\nRecent faces:")
        for row in rows:
            print(row)
            # Check if file exists
            img_path = row[2]
            print(f"   Image Path stored: {img_path}")
            if os.path.exists(img_path):
                 print("   ✅ Image file exists.")
            else:
                 print("   ❌ Image file NOT found on disk.")
                 
    conn.close()
    
except Exception as e:
    print(f"Error accessing database: {e}")
