import cv2
import pickle
import os
import threading
import numpy as np
import easyocr
import time
import queue
import re
import pygame
import sqlite3
from ultralytics import YOLO
from gtts import gTTS
# from deepface import DeepFace  # Keeping disabled for RPi speed, unless needed

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Translations
TRANSLATION_DICT = {
    "person": "kişi", "cell phone": "telefon", "book": "kitap",
    "laptop": "bilgisayar", "mouse": "fare", "keyboard": "klavye",
    "chair": "sandalye", "bottle": "şişe", "cup": "bardak",
    "table": "masa", "pen": "kalem", "tv": "televizyon",
    "remote": "kumanda", "spoon": "kaşık", "fork": "çatal",
    "knife": "bıçak"
}

# Static Audio Cache
AUDIO_CACHE = {
    "scanning": "cache_scanning.mp3",
    "currency_scan": "cache_currency_scan.mp3",
    "color_scan": "cache_color_scan.mp3",
    "no_text": "cache_no_text.mp3",
    "no_money": "cache_no_money.mp3",
    "error": "cache_error.mp3",
    "too_short": "cache_too_short.mp3",
    "no_face": "cache_no_face.mp3"
}

class VoiceManager:
    """
    Handles text-to-speech with caching for static messages to improve speed.
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.is_speaking = False
        self.audio_enabled = False
        try:
            # Initialize pygame mixer with specific settings for better compatibility
            pygame.mixer.quit()  # Ensure clean state
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_enabled = True
            print("Audio System: ENABLED")
        except Exception as e:
            print(f"Audio Init Failed: {e}")
            print("Warning: Audio will not work!")
        self._ensure_cache()
        threading.Thread(target=self._worker, daemon=True).start()

    def _ensure_cache(self):
        """Pre-generates common audio files to avoid network lag."""
        if not os.path.exists("audio_cache"):
            os.makedirs("audio_cache")
        
        messages = {
            "scanning": "Okuyorum...",
            "currency_scan": "Parayı kontrol ediyorum...",
            "color_scan": "Rengi kontrol ediyorum...",
            "no_text": "Yazı göremedim.",
            "no_money": "Para tanıyamadım.",
            "error": "Bir hata oluştu.",
            "too_short": "Yazı anlaşılamadı.",
            "no_face": "Tanıdık kimse yok."
        }
        
        print("🔊 Checking audio cache...")
        for key, text in messages.items():
            path = os.path.join("audio_cache", f"{key}.mp3")
            if not os.path.exists(path):
                print(f"   Generating: {key}...")
                try:
                    tts = gTTS(text=text, lang='tr', slow=False)
                    tts.save(path)
                except Exception as e:
                    print(f"   ⚠️ Could not cache {key}: {e}")
            AUDIO_CACHE[key] = path

    def speak(self, text, lang='tr', use_cache_key=None):
        """Adds text to the speech queue. Priority to cached files."""
        if not text and not use_cache_key:
            return
        
        # If it's a static message, prioritize it
        if use_cache_key and use_cache_key in AUDIO_CACHE:
            path = AUDIO_CACHE[use_cache_key]
            if os.path.exists(path):
                print(f"🗣️  Playing Cached: {use_cache_key}")
                self.queue.put(("__FILE__", path))
                return

        print(f"🗣️  Queueing: {text}")
        self.queue.put((text, lang))

    def _worker(self):
        while True:
            data, lang = self.queue.get()
            if data:
                self.is_speaking = True
                try:
                    if not self.audio_enabled:
                        print("Audio disabled - skipping playback")
                        self.queue.task_done()
                        continue
                        
                    file_to_play = None
                    if data == "__FILE__":
                        # Play existing file
                        file_to_play = lang
                        print(f"Playing cached audio: {file_to_play}")
                    else:
                        # Generate new file
                        filename = f"temp_speech_{int(time.time())}.mp3"
                        print(f"Generating speech: {data[:50]}...")
                        tts = gTTS(text=data, lang=lang, slow=False)
                        tts.save(filename)
                        file_to_play = filename
                        print(f"Speech saved to: {filename}")
                    
                    if file_to_play and os.path.exists(file_to_play):
                        print(f"Loading audio file: {file_to_play}")
                        pygame.mixer.music.load(file_to_play)
                        pygame.mixer.music.set_volume(1.0)  # Ensure volume is at max
                        print("Playing audio...")
                        pygame.mixer.music.play()
                        
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        pygame.mixer.music.unload() # Unlock file
                        print("Audio playback completed")
                        
                        # Cleanup temp files
                        if file_to_play.startswith("temp_speech_"):
                            try:
                                os.remove(file_to_play)
                            except:
                                pass
                    else:
                        print(f"ERROR: Audio file not found: {file_to_play}")
                except Exception as e:
                    print(f"Sound Error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.is_speaking = False
            self.queue.task_done()

class CurrencyMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.templates = []
        self._load_templates()
        
        # FLANN Matcher Settings
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _load_templates(self):
        template_dir = "currency_templates"
        if not os.path.exists(template_dir):
            print("⚠️ Warning: 'currency_templates' folder not found.")
            return

        print("💰 Loading Currency Templates...")
        for f in os.listdir(template_dir):
            if f.endswith(".jpg"):
                try:
                    path = os.path.join(template_dir, f)
                    img = cv2.imread(path, 0) # Read as gray
                    if img is None: continue
                    
                    # Compute descriptors once at startup
                    kp, des = self.sift.detectAndCompute(img, None)
                    if des is not None:
                        val = os.path.splitext(f)[0] # e.g. "200"
                        self.templates.append((val, kp, des))
                        print(f"   ✅ Loaded: {val} TL")
                except Exception as e:
                    print(f"   ❌ Error loading {f}: {e}")

    def match(self, target_gray_img):
        """
        Returns the value string (e.g. "200") if a match is found, else None.
        """
        # Find features in the target image
        kp_target, des_target = self.sift.detectAndCompute(target_gray_img, None)
        
        if des_target is None or len(des_target) < 10:
            return None
            
        best_match_val = None
        max_good_matches = 0
        
        for val, kp_temp, des_temp in self.templates:
            try:
                matches = self.matcher.knnMatch(des_temp, des_target, k=2)
                
                # Lowe's Ratio Test
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                
                if len(good) > 10: # Minimum matches threshold
                    # Geometric Verification (Homography) - The Gold Standard
                    src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        matchesMask = mask.ravel().tolist()
                        inliers = sum(matchesMask)
                        
                        # We need a significant number of INLIERS (points that fit the plane)
                        # Threshold can be tuned. 10-15 is usually very safe for distinct objects.
                        if inliers > 12: 
                            if inliers > max_good_matches:
                                max_good_matches = inliers
                                best_match_val = val
            except:
                continue
                
        return best_match_val

class ColorDetector:
    def __init__(self):
        # 1. Try to load Custom YOLO Model (Hybrid Mode)
        self.model = None
        self.model_path = "runs/classify/color_model_custom/weights/best.pt"
        
        if os.path.exists(self.model_path):
            print(f"🎨 CUSTOM AI COLOR MODEL FOUND! Loading: {self.model_path}")
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"⚠️ Error loading custom model: {e}")
        else:
            print("🎨 No custom AI color model found. Using basic HSV logic.")
            print("   (Run 'capture_colors.py' then 'train_colors.py' to enable AI mode)")

        # 2. Fallback: Standard HSV Colors
        # Format: Name -> (Lower HSV, Upper HSV)
        self.colors = [
            ("Kırmızı", [0, 100, 100], [10, 255, 255]),
            ("Kırmızı", [160, 100, 100], [180, 255, 255]), 
            ("Turuncu", [10, 100, 100], [25, 255, 255]),
            ("Sarı", [25, 100, 100], [35, 255, 255]),
            ("Yeşil", [35, 100, 100], [85, 255, 255]),
            ("Mavi", [85, 100, 100], [125, 255, 255]),
            ("Mor", [125, 100, 100], [150, 255, 255]),
            ("Siyah", [0, 0, 0], [180, 255, 50]),    
            ("Beyaz", [0, 0, 200], [180, 30, 255]),  
            ("Gri", [0, 0, 50], [180, 50, 200])      
        ]

    def get_color_name(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # --- A. AI Model Prediction (If Available) ---
        if self.model:
            # Crop a larger center square for the AI (224x224 is ideal for YOLO Classification)
            crop_size = 224
            half = crop_size // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            
            roi_ai = frame[y1:y2, x1:x2]
            
            if roi_ai.size > 0:
                try:
                    # Run inference
                    results = self.model.predict(roi_ai, verbose=False, conf=0.6)
                    # Check confidence
                    if results and results[0].probs.top1conf.item() > 0.6:
                        color_name = results[0].names[results[0].probs.top1]
                        # Translations or corrections can be added here if needed
                        return color_name 
                except:
                    pass # Fallback to HSV if inference fails

        # --- B. HSV Logic (Fallback) ---
        # Take a small 20x20 sample from center
        roi = frame[cy-10:cy+10, cx-10:cx+10]
        if roi.size == 0: return "Bilinmeyen"
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_roi, axis=(0, 1))
        h_val, s_val, v_val = mean_hsv
        
        for name, lower, upper in self.colors:
            if (lower[0] <= h_val <= upper[0]) and \
               (lower[1] <= s_val <= upper[1]) and \
               (lower[2] <= v_val <= upper[2]):
                return name
                
        return "Bilinmeyen"

class SmartGlassesAssistant:
    def __init__(self):
        print("\n🚀 Initializing Smart Glasses System (DeepFace + YOLOv8 Nano)...")
        
        self.voice = VoiceManager()
        
        print("1️⃣  Loading OCR (Reader - Turkish Optimized)...")
        # Put 'tr' first for priority
        self.reader = easyocr.Reader(['tr', 'en'], gpu=False) 

        print("2️⃣  Loading YOLO (Nano for RPi 5 Speed)...")
        # SWITCHED TO NANO FOR RASPBERRY PI 5
        self.model = YOLO('yolov8n.pt') 

        # Try to load custom currency model (Classification)
        self.currency_model = None
        # Use the advanced model with 100% accuracy
        custom_model_path = "runs/classify/currency_cls_advanced/weights/best.pt"
        if os.path.exists(custom_model_path):
            print(f"🌟 Loading Advanced Currency Classifier: {custom_model_path}")
            print(f"   ✨ Model Accuracy: 100% (50 epochs training)")
            self.currency_model = YOLO(custom_model_path)
        else:
            # Fallback to basic model
            basic_model_path = "runs/classify/currency_cls/weights/best.pt"
            if os.path.exists(basic_model_path):
                print(f"⚠️  Using Basic Currency Classifier: {basic_model_path}")
                self.currency_model = YOLO(basic_model_path)
            else:
                print("ℹ️  No Custom Currency Model found. Using SIFT only.") 

        print("3️⃣  Loading Known Faces (DeepFace Facenet512)...")
        # DeepFace Import inside try/except in case of heavy load
        try:
             from deepface import DeepFace
        except:
             print("⚠️ DeepFace module import issue.")
        
        self.known_encodings = []
        self.known_names = []
        
        # Load Haar Cascade for fast detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
             print("⚠️ Warning: Could not load Haar Cascade.")
             self.face_cascade = None

        self._load_faces()
        
        print("4️⃣  Loading Currency Matcher (SIFT)...")
        self.currency_matcher = CurrencyMatcher()

        print("5️⃣  Loading Color Detector (Hybrid AI/HSV)...")
        self.color_detector = ColorDetector()

        self.cap = None
        self.last_announced = {}

    def _load_faces(self):
        """
        Loads Face Embeddings directly from the Web App's SQLite Database.
        This ensures 100% synchronization with the web interface.
        """
        print("   📂 Loading faces from Database...")
        
        # Path to the Web App's database
        db_path = os.path.join("web_face_recognition", "instance", "face_recognition.db")
        
        if not os.path.exists(db_path):
            print(f"      ⚠️ Database not found at: {db_path}")
            print("      ⚠️ Please ensure the Web App is set up and faces are added.")
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face'")
            if not cursor.fetchone():
                print("      ⚠️ Table 'face' not found in database.")
                conn.close()
                return

            # Query names and encodings
            cursor.execute("SELECT name, encoding FROM face")
            rows = cursor.fetchall()
            
            self.known_encodings = []
            self.known_names = []
            count = 0
            
            for name, encoding_blob in rows:
                if encoding_blob:
                    try:
                        # Load embedding from binary blob
                        embedding = pickle.loads(encoding_blob)
                        self.known_encodings.append(embedding)
                        self.known_names.append(name)
                        print(f"      👤 Loaded (DB): {name}")
                        count += 1
                    except Exception as e:
                        print(f"      ❌ Error parsing embedding for {name}: {e}")
            
            print(f"      ✅ Total Database Faces: {count}")
            conn.close()
            
        except Exception as e:
            print(f"      ❌ Database connection error: {e}")

    def cosine_similarity(self, vec1, vec2):
        """Calculates Cosine Similarity between two vectors"""
        try:
            vec1 = np.array(vec1).flatten()
            vec2 = np.array(vec2).flatten()
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def detect_language(self, text):
        """
        Simple heuristic to detect Turkish vs English.
        """
        tr_chars = "ğüşıöçĞÜŞİÖÇ"
        for char in tr_chars:
            if char in text:
                return 'tr'
        return 'en'

    def preprocess_for_ocr(self, frame):
        """
        Enhances image for better text recognition.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Removed sharpening to prevent character fragmentation
        return gray

    def describe_scene(self, frame, results, face_names):
        """
        Generates a verbal description of the scene.
        """
        detected_objects = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.4: 
                    en_name = self.model.names[cls_id]
                    tr_name = TRANSLATION_DICT.get(en_name, en_name)
                    detected_objects.append(tr_name)

        msg_parts = []
        if face_names:
            unique_faces = list(set(face_names))
            if "Bilinmeyen" in unique_faces and len(unique_faces) > 1:
                unique_faces.remove("Bilinmeyen")
            msg_parts.append(f"{', '.join(unique_faces)} görüyorum.")

        if detected_objects:
            from collections import Counter
            counts = Counter(detected_objects)
            obj_desc = []
            for name, count in counts.most_common(3):
                obj_desc.append(f"{count} {name}")
            msg_parts.append(f"Burada {', '.join(obj_desc)} var.")

        if not msg_parts:
            return "Hiçbir şey göremiyorum."
        return " ".join(msg_parts)
    
    def identify_currency(self, frame):
        """
        Analyzes frame for Turkish Banknotes.
        Priority: Custom YOLO Classifier -> SIFT Feature Matching
        """
        self.voice.speak(None, use_cache_key="currency_scan")
        
        # --- Option A: Custom Deep Learning Classifier (Best) ---
        if self.currency_model:
            print("💰 Scanning with Custom Classifier...")
            # Predict
            results = self.currency_model.predict(frame, verbose=False)
            
            if results and len(results) > 0:
                probs = results[0].probs
                top1_index = probs.top1
                conf = probs.top1conf.item()
                
                # DEBUG: Print top 3 predictions
                print(f"   🔍 Top 1: {results[0].names[probs.top1]} ({probs.top1conf.item():.2f})")
                if hasattr(probs, 'top5'):
                   # Print next 2 if available
                   for i in range(1, min(3, len(probs.top5))):
                       idx = probs.top5[i]
                       print(f"      Alt {i}: {results[0].names[idx]} ({probs.data[idx]:.2f})")

                # Lower threshold slightly and fix logic
                if conf > 0.50:
                    cls_name = results[0].names[top1_index] 
                    msg = f"{cls_name} Türk Lirası"
                    print(f"💰 AI RESULT: {msg} ({conf:.2f})")
                    self.voice.speak(msg)
                    return

            # If confident check fails, we could fallback or just say nothing
            # Let's fallback to SIFT just in case
        
        # --- Option B: SIFT Feature Matching (Fallback) ---
        print("Fallback to SIFT...")
        
        # 1. Use full resolution for better distance detail
        # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Check Normal
        val = self.currency_matcher.match(gray)
        
        # 3. Check Mirrored (Flip Horizontal) if no match
        if not val:
           mirrored_gray = cv2.flip(gray, 1)
           val = self.currency_matcher.match(mirrored_gray)
        
        if val:
             msg = f"{val} Türk Lirası"
             print(f"RESULT: {msg}")
             self.voice.speak(msg)
        else:
             self.voice.speak(None, use_cache_key="no_money")

    def identify_color(self, frame):
        """
        Uses ColorDetector to identify the color in the center of the frame.
        """
        self.voice.speak(None, use_cache_key="color_scan")
        color_name = self.color_detector.get_color_name(frame)
        print(f"🎨 Color Detected: {color_name}")
        
        if color_name != "Bilinmeyen":
            self.voice.speak(f"Bu {color_name}")
        else:
            self.voice.speak("Rengi anlayamadım")

    def read_text(self, frame):
        """
        Reads text from the center of the frame (Faster & More Accurate).
        """
        self.voice.speak(None, use_cache_key="scanning")
        print("📖 Scanning text (Center Zone)...")
        
        try:
            h, w = frame.shape[:2]
            start_x, start_y = int(w * 0.2), int(h * 0.3)
            end_x, end_y = int(w * 0.8), int(h * 0.7)
            
            roi = frame[start_y:end_y, start_x:end_x]
            processed_roi = self.preprocess_for_ocr(roi)
            # x_ths=2.0 helps merge words horizontally
            # mag_ratio=1.5 enlarges text for better recognition
            results = self.reader.readtext(processed_roi, detail=0, paragraph=True, x_ths=2.0, mag_ratio=1.5)
            
            if results:
                full_text = " ".join(results)
                
                full_text = full_text.title() 

                if re.search(r'[a-zA-ZğüşıöçĞÜŞİÖÇ]', full_text):
                    print(f"📜 READ: {full_text}")
                    # FORCE TURKISH as requested
                    self.voice.speak(f"{full_text}", lang='tr')
                else:
                    self.voice.speak(None, use_cache_key="too_short")
            else:
                self.voice.speak(None, use_cache_key="no_text")

        except Exception as e:
            print(f"❌ OCR Error: {e}")
            self.voice.speak(None, use_cache_key="error")

    def run(self):
        self.cap = cv2.VideoCapture(0)
        try:
            # OPTIMIZATION: Lower resolution for RPi 5 Speed (640x480)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
        except:
            pass
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera.")
            return

        print("\n🚀 SYSTEM READY!")
        print("👉 Press 'Space' to describe scene.")
        print("👉 Press 'r' to read text.")
        print("👉 Press 'c' to scan CURRENCY (New!).")
        print("👉 Press 'x' to detect COLOR (New!).")
        print("👉 Press 'f' to identify FACES (New!).")
        print("👉 Press 'q' to quit.\n")

        frame_count = 0
        current_faces = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            frame_count += 1

            # --- Reload Faces Periodically (Every 300 frames ~ 10-15 sec) ---
            if frame_count % 300 == 0:
                print("🔄 Syncing with Database...")
                self._load_faces()
            
            # --- Object Detection ---
            if frame_count % 3 == 0:
                # OPTIMIZATION: Use imgsz=320 for speed if needed, or keeping it default 640
                results = self.model(frame, verbose=False, conf=0.35)
                self.last_results = results
            
            # Custom Drawing for Turkish Labels
            if hasattr(self, 'last_results'):
                # Draw boxes manually
                for r in self.last_results:
                    for box in r.boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Label
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        en_name = self.model.names[cls_id]
                        tr_name = TRANSLATION_DICT.get(en_name, en_name) # Fallback to English if not found
                        
                        label = f"{tr_name} {conf:.2f}"
                        # Text Background
                        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w_text, y1), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- Face Recognition (DeepFace Facenet512) ---
            # Run every 5 frames to balance speed
            if frame_count % 5 == 0:
                # Import DeepFace here to allow lazy loading
                from deepface import DeepFace

                if self.face_cascade and len(self.known_names) > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Detect faces using Haar Cascade (Fast)
                    faces_detected = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    current_faces_temp = [] # Temp list to avoid flickering empty
                    
                    found_face = False
                    for (x, y, w, h) in faces_detected:
                        found_face = True
                        # Draw bounding box for every detected face
                        # Start with Unknown color
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        
                        try:
                            # 1. Crop Face
                            face_roi = frame[y:y+h, x:x+w]
                            if face_roi.size == 0: continue
                            
                            # 2. Get Embedding (Facenet512)
                            # enforce_detection=False because we already cropped the face
                            embedding_objs = DeepFace.represent(
                                img_path=face_roi,
                                model_name='Facenet512',
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            if embedding_objs and len(embedding_objs) > 0:
                                current_embedding = embedding_objs[0]['embedding']
                                
                                # 3. Compare with known faces using Cosine Similarity
                                best_match_score = -1.0
                                best_match_name = "Bilinmeyen"
                                
                                for idx, known_enc in enumerate(self.known_encodings):
                                    score = self.cosine_similarity(current_embedding, known_enc)
                                    if score > best_match_score:
                                        best_match_score = score
                                        best_match_name = self.known_names[idx]
                                
                                # 4. Threshold (Around 0.4 for Facenet512 usually, but Cosine is usually higher is better)
                                # DeepFace cosine distance is usually (1 - cosine_similarity).
                                # But here we calculated explicit cosine similarity (-1 to 1).
                                # Good match is usually > 0.4 or 0.5 depending on the model.
                                # Let's try 0.4
                                if best_match_score > 0.4:
                                    # print(f"Face Match: {best_match_name} ({best_match_score:.2f})")
                                    current_faces_temp.append(best_match_name)
                                    
                                    # Update UI
                                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(display_frame, f"{best_match_name} ({best_match_score:.2f})", (x, y-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    # Auto-Announce REMOVED (User Request)
                                    # Only speaks on 'F' or 'Space' key press now.

                        except Exception as e:
                            # print(f"DEBUG: DeepFace Error: {e}")
                            pass
                    
                    if found_face:
                         current_faces = current_faces_temp

            # --- Draw Reading/Currency/Color Zone ---
            # Center of the screen
            start_x, start_y = int(w * 0.2), int(h * 0.3)
            end_x, end_y = int(w * 0.8), int(h * 0.7)
            cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            
            # --- Draw Color Center Point ---
            cx, cy = w // 2, h // 2
            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1) # Red dot for color target
            
            # Overlay instructions
            overlay_text = "R:Read C:Cash X:Color F:Face Space:Scan"
            cv2.putText(display_frame, overlay_text, (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Display Faces (List on left) ---
            y_offset = 50
            for person in current_faces:
                cv2.putText(display_frame, f"Face: {person}", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 35

            cv2.imshow("Smart Glasses Pro", display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            # Case Insensitive Checks
            if key in [ord('q'), ord('Q')]:
                break
            elif key == ord(' '): # Describe (Space)
                if hasattr(self, 'last_results'):
                    desc = self.describe_scene(frame, self.last_results, current_faces)
                    print(f"🤖 AI: {desc}")
                    self.voice.speak(desc)
            elif key in [ord('r'), ord('R')]: # Read
                self.read_text(frame)
            elif key in [ord('c'), ord('C')]: # Currency
                self.identify_currency(frame)
            elif key in [ord('x'), ord('X')]: # Color
                self.identify_color(frame)
            elif key in [ord('f'), ord('F')]: # Face Identity (Manual)
                print("👤 Manual Face Check...")
                if current_faces:
                    names = ", ".join(current_faces)
                    self.voice.speak(f"Bu {names}")
                else:
                    self.voice.speak("Tanıdık kimse yok.")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SmartGlassesAssistant()
    app.run()
