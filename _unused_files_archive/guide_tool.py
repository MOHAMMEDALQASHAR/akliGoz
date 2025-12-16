import cv2
import os

# Folder name where faces will be stored
folder = "known_faces"

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Start the camera
cap = cv2.VideoCapture(0)

print("\n--- 📸 Face Registration Tool (Guide App) ---")
print("1. Point the camera at the person's face.")
print("2. Press 's' to capture and save.")
print("3. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access the camera.")
        break
    
    # Show the video feed
    cv2.imshow("Guide Tool - Add Face", frame)
    
    key = cv2.waitKey(1)
    
    # If 's' is pressed (Save)
    if key == ord('s'):
        # Pause to take input from terminal
        print("\n[PAUSED] Switch to terminal to type the name...")
        name = input("✍️  Enter person's name (English): ")
        
        if name:
            # Create filename and save
            filename = f"{folder}/{name}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Saved successfully: {name}.jpg")
            print("--> You can add another person or press 'q' to quit.")
        else:
            print("⚠️ No name entered! Please try again.")
            
    # If 'q' is pressed (Quit)
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
