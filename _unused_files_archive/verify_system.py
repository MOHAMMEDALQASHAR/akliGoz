
import os
import sys

# Add the web_face_recognition directory to path so we can import app
sys.path.append(os.path.join(os.getcwd(), 'web_face_recognition'))

from app import app, db, Face, User

print("--- SYSTEM VERIFICATION ---")

# 1. Verify Paths
print(f"App Instance Path: {app.instance_path}")
print(f"Config DB URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

expected_db_path = os.path.join(os.getcwd(), 'web_face_recognition', 'instance', 'face_recognition.db')
print(f"Expected DB File: {expected_db_path}")

if os.path.exists(expected_db_path):
    print("YES Database file exists on disk.")
else:
    print("NO Database file DOES NOT exist.")

# 2. Try Database Connection & Write
print("\n--- Testing Database Write ---")
try:
    with app.app_context():
        # Create user if not exists
        user = User.query.first()
        if not user:
            print("Creating dummy user for test...")
            user = User(email='test@test.com', name='Test User', password='hashedpassword')
            db.session.add(user)
            db.session.commit()
            print(f"Created User ID: {user.id}")
        else:
            print(f"Found existing User ID: {user.id}")

        # Try to add a face
        print("Attempting to add a test face record...")
        try:
            test_face = Face(
                user_id=user.id,
                name="System Test Face",
                image_path="test_path.jpg",
                encoding=b'fake_encoding'
            )
            db.session.add(test_face)
            db.session.commit()
            print(f"SUCCESS Successfully added Face ID: {test_face.id}")
            
            # Verify read
            f = Face.query.get(test_face.id)
            if f:
                print(f"SUCCESS Verified Read: {f.name}")
                
                # Cleanup
                db.session.delete(test_face)
                db.session.commit()
                print("SUCCESS Cleanup successful (deleted test face)")
            else:
                print("FAILED Could not read back the face!")
                
        except Exception as e:
            print(f"FAILED Failed to write face: {e}")
            db.session.rollback()

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
