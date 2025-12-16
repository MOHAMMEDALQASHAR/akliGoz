import sqlite3
import os

db_path = os.path.join(os.getcwd(), 'web_face_recognition', 'instance', 'face_recognition.db')
print(f"Checking database: {db_path}")

if not os.path.exists(db_path):
    print("ERROR: Database file does not exist!")
    exit(1)

print("Database file exists.")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check users
cursor.execute("SELECT id, name, email FROM user")
users = cursor.fetchall()
print(f"\n--- USERS ({len(users)}) ---")
for u in users:
    print(f"  ID: {u[0]}, Name: {u[1]}, Email: {u[2]}")

# Check faces
cursor.execute("SELECT id, user_id, name, image_path FROM face")
faces = cursor.fetchall()
print(f"\n--- FACES ({len(faces)}) ---")
for f in faces:
    print(f"  ID: {f[0]}, User: {f[1]}, Name: {f[2]}")
    print(f"    Image: {f[3]}")
    if os.path.exists(f[3]):
        print(f"    [File EXISTS]")
    else:
        print(f"    [File MISSING]")

conn.close()
