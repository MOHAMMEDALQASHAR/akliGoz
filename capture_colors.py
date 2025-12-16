import cv2
import os
import time
import pygame
from gtts import gTTS

# --- Configuration ---
DATASET_DIR = "datasets/colors"
COLORS = [
    "Kirmizi", "Turuncu", "Sari", "Yesil", "Mavi", 
    "Mor", "Siyah", "Beyaz", "Gri"
]
IMG_SIZE = 224

# Initialize Audio
try:
    pygame.mixer.init()
except:
    pass

def speak(text):
    print(f"🗣️ AI: {text}")
    try:
        tts = gTTS(text=text, lang='tr')
        filename = "temp_capture_cmd.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        try:
            os.remove(filename)
        except:
            pass
    except Exception as e:
        print(f"TTS Error: {e}")

def setup_dirs():
    for color in COLORS:
        os.makedirs(f"{DATASET_DIR}/train/{color}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/val/{color}", exist_ok=True)

def main():
    setup_dirs()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("\n🎨 COLOR TRAINING DATA COMPANION (VOICE ENABLED) 🎨")
    speak("Renk eğitimi başlıyor. Lütfen renkleri hazırlayın.")

    for color_name in COLORS:
        print(f"\n👉 Next Color: {color_name.upper()}")
        
        # Voice Prompt
        speak(f"Lütfen {color_name} rengini gösterin.")
        time.sleep(1) # Give time to react
        speak("3... 2... 1... Başlıyor.")
        time.sleep(0.5)
        
        # Capture Loop
        count = 0
        max_count = 100
        
        while count < max_count:
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            cy, cx = h // 2, w // 2
            half = IMG_SIZE // 2
            
            # Draw Box
            cv2.rectangle(frame, (cx-half, cy-half), (cx+half, cy+half), (0, 255, 0), 2)
            cv2.putText(frame, f"{color_name}: {count}/{max_count}", (cx-half, cy-half-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Crop Center
            # Ensure crop boundaries are valid
            y1, y2 = max(0, cy-half), min(h, cy+half)
            x1, x2 = max(0, cx-half), min(w, cx+half)
            
            crop = frame[y1:y2, x1:x2]
            
            # Save
            import random
            split = "train" if random.random() > 0.2 else "val" # 80/20 Split
            
            timestamp = int(time.time() * 1000)
            filename = f"{DATASET_DIR}/{split}/{color_name}/{timestamp}.jpg"
            
            if crop.size > 0:
                cv2.imwrite(filename, crop)
                count += 1
            
            # Visual Feedback
            cv2.imshow("Color Collector", frame)
            cv2.waitKey(20) # Capture speed (adjust if too fast)
            
        speak(f"{color_name} tamamlandı.")
        # Pause briefly between colors
        time.sleep(1.5)

    speak("Tüm renkler tamamlandı. Şimdi eğitimi başlatabilirsiniz.")
    print("\n✅ All Colors Captured!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
