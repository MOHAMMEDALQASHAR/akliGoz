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

# --------------------------------------------------------------------------------
# ⚠️ THIS FILE IS SPECIALLY PREPARED FOR RASPBERRY PI 5 + AI KIT (HAILO-8L)
# ⚠️ DO NOT RUN ON WINDOWS. IT EXPECTS .HEF FILES
# --------------------------------------------------------------------------------

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
    def __init__(self):
        self.queue = queue.Queue()
        self.is_speaking = False
        self.audio_enabled = False
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_enabled = True
        except Exception as e:
            print(f"Audio Init Failed: {e}")
        self._ensure_cache()
        threading.Thread(target=self._worker, daemon=True).start()

    def _ensure_cache(self):
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
        for key, text in messages.items():
            path = os.path.join("audio_cache", f"{key}.mp3")
            if not os.path.exists(path):
                try:
                    tts = gTTS(text=text, lang='tr', slow=False)
                    tts.save(path)
                except: pass
            AUDIO_CACHE[key] = path

    def speak(self, text, lang='tr', use_cache_key=None):
        if not text and not use_cache_key: return
        if use_cache_key and use_cache_key in AUDIO_CACHE:
            path = AUDIO_CACHE[use_cache_key]
            if os.path.exists(path):
                self.queue.put(("__FILE__", path))
                return
        self.queue.put((text, lang))

    def _worker(self):
        while True:
            data, lang = self.queue.get()
            if data:
                self.is_speaking = True
                try:
                    if self.audio_enabled:
                        file_to_play = None
                        if data == "__FILE__":
                            file_to_play = lang
                        else:
                            filename = f"temp_speech_{int(time.time())}.mp3"
                            tts = gTTS(text=data, lang=lang, slow=False)
                            tts.save(filename)
                            file_to_play = filename
                        
                        if file_to_play and os.path.exists(file_to_play):
                            pygame.mixer.music.load(file_to_play)
                            pygame.mixer.music.set_volume(1.0)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            pygame.mixer.music.unload()
                            if file_to_play.startswith("temp_speech_"):
                                try: os.remove(file_to_play)
                                except: pass
                except Exception as e:
                    print(f"Sound Error: {e}")
                finally:
                    self.is_speaking = False
            self.queue.task_done()

class CurrencyMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.templates = []
        self._load_templates()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _load_templates(self):
        template_dir = "currency_templates"
        if os.path.exists(template_dir):
            for f in os.listdir(template_dir):
                if f.endswith(".jpg"):
                    try:
                        path = os.path.join(template_dir, f)
                        img = cv2.imread(path, 0)
                        if img is not None:
                            kp, des = self.sift.detectAndCompute(img, None)
                            if des is not None:
                                self.templates.append((os.path.splitext(f)[0], kp, des))
                    except: pass

    def match(self, target_gray_img):
        kp_target, des_target = self.sift.detectAndCompute(target_gray_img, None)
        if des_target is None or len(des_target) < 10: return None
        best_match_val = None
        max_good_matches = 0
        for val, kp_temp, des_temp in self.templates:
            try:
                matches = self.matcher.knnMatch(des_temp, des_target, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]
                if len(good) > 10:
                    src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        inliers = sum(mask.ravel().tolist())
                        if inliers > 12 and inliers > max_good_matches:
                            max_good_matches = inliers
                            best_match_val = val
            except: continue
        return best_match_val

class ColorDetector:
    def __init__(self):
        # HAILO MODE: Try to load HEF
        self.model = None
        self.model_path = "color_model.hef" # <--- CHANGED FOR HAILO
        
        if os.path.exists(self.model_path):
            print(f"🎨 Custom Hailo Color Model Found!")
            try:
                self.model = YOLO(self.model_path)
            except: pass
        
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
        # AI Logic
        if self.model:
            crop_size = 224
            half = crop_size // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            roi_ai = frame[y1:y2, x1:x2]
            if roi_ai.size > 0:
                try:
                    results = self.model.predict(roi_ai, verbose=False, conf=0.6)
                    if results and results[0].probs.top1conf.item() > 0.6:
                        return results[0].names[results[0].probs.top1]
                except: pass
        # HSV Fallback
        roi = frame[cy-10:cy+10, cx-10:cx+10]
        if roi.size == 0: return "Bilinmeyen"
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_roi, axis=(0, 1))
        h, s, v = mean_hsv
        for name, lower, upper in self.colors:
            if (lower[0] <= h <= upper[0]) and (lower[1] <= s <= upper[1]) and (lower[2] <= v <= upper[2]):
                return name
        return "Bilinmeyen"

class SmartGlassesAssistant:
    def __init__(self):
        print("\n🚀 Initializing Smart Glasses (HAILO AI EDITION)...")
        self.voice = VoiceManager()
        self.reader = easyocr.Reader(['tr', 'en'], gpu=False) 

        print("2️⃣  Loading YOLO (Hailo-8L)...")
        # -----------------------------------------------------------
        # HERE IS THE MAGIC SWITCH FOR RASPBERRY PI AI KIT
        # -----------------------------------------------------------
        try:
            # On RPi with Ultralytics + Hailo, we use the .hef file
            self.model = YOLO('yolov8n.hef') 
            print("   ✅ Loaded yolov8n.hef (Optimized)")
        except Exception as e:
            print(f"   ❌ Error loading YOLO HEF: {e}")
            print("   ⚠️  Using standard YOLO CPU fallback (Slow!)")
            self.model = YOLO('yolov8n.pt')

        print("3️⃣  Loading Currency (Hailo)...")
        self.currency_model = None
        if os.path.exists("currency_model.hef"):
            try:
                self.currency_model = YOLO("currency_model.hef")
                print("   ✅ Loaded currency_model.hef")
            except: pass
        else:
             print("   INFO: currency_model.hef not found. (Using SIFT)")

        # DeepFace (CPU)
        try: from deepface import DeepFace
        except: pass
        self.known_encodings = []
        self.known_names = []
        try: self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except: self.face_cascade = None
        self._load_faces()
        
        self.currency_matcher = CurrencyMatcher()
        self.color_detector = ColorDetector()
        self.cap = None

    def _load_faces(self):
        # ... Same DB logic ...
        db_path = os.path.join("web_face_recognition", "instance", "face_recognition.db")
        if not os.path.exists(db_path): return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, encoding FROM face")
            rows = cursor.fetchall()
            self.known_encodings = []
            self.known_names = []
            for name, encoding_blob in rows:
                if encoding_blob:
                    try:
                        self.known_encodings.append(pickle.loads(encoding_blob))
                        self.known_names.append(name)
                    except: pass
            conn.close()
        except: pass

    def cosine_similarity(self, vec1, vec2):
        try:
            vec1 = np.array(vec1).flatten()
            vec2 = np.array(vec2).flatten()
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except: return 0.0

    def preprocess_for_ocr(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def describe_scene(self, frame, results, face_names):
        detected_objects = []
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.4: 
                    en = self.model.names[int(box.cls[0])]
                    detected_objects.append(TRANSLATION_DICT.get(en, en))
        msg_parts = []
        if face_names:
            u = list(set(face_names))
            if "Bilinmeyen" in u and len(u) > 1: u.remove("Bilinmeyen")
            msg_parts.append(f"{', '.join(u)} görüyorum.")
        if detected_objects:
            from collections import Counter
            c = Counter(detected_objects)
            obj_desc = [f"{v} {k}" for k, v in c.most_common(3)]
            msg_parts.append(f"Burada {', '.join(obj_desc)} var.")
        return " ".join(msg_parts) if msg_parts else "Hiçbir şey göremiyorum."

    def identify_currency(self, frame):
        self.voice.speak(None, use_cache_key="currency_scan")
        if self.currency_model:
            results = self.currency_model.predict(frame, verbose=False)
            if results and results[0].probs.top1conf.item() > 0.50:
                cls_name = results[0].names[results[0].probs.top1]
                self.voice.speak(f"{cls_name} Türk Lirası")
                return
        # Fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        val = self.currency_matcher.match(gray)
        if not val: val = self.currency_matcher.match(cv2.flip(gray, 1))
        if val: self.voice.speak(f"{val} Türk Lirası")
        else: self.voice.speak(None, use_cache_key="no_money")

    def identify_color(self, frame):
        self.voice.speak(None, use_cache_key="color_scan")
        name = self.color_detector.get_color_name(frame)
        if name != "Bilinmeyen": self.voice.speak(f"Bu {name}")
        else: self.voice.speak("Rengi anlayamadım")

    def read_text(self, frame):
        self.voice.speak(None, use_cache_key="scanning")
        try:
            h, w = frame.shape[:2]
            roi = frame[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
            results = self.reader.readtext(self.preprocess_for_ocr(roi), detail=0, paragraph=True, x_ths=2.0, mag_ratio=1.5)
            if results:
                txt = " ".join(results).title()
                if re.search(r'[a-zA-ZğüşıöçĞÜŞİÖÇ]', txt): self.voice.speak(txt, lang='tr')
                else: self.voice.speak(None, use_cache_key="too_short")
            else: self.voice.speak(None, use_cache_key="no_text")
        except: self.voice.speak(None, use_cache_key="error")

    def run(self):
        # HAILO OPTIMIZATION: 640x640 is usually preferred for YOLOv8
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640); self.cap.set(4, 480)
        
        print("\n🚀 RASPBERRY PI 5 AI SYSTEM READY!")
        print("CONFIRM: Using specific keys... F:Face, X:Color, C:Cash")

        frame_count = 0
        current_faces = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera. Exiting...")
                break
            display = frame.copy()
            frame_count += 1
            if frame_count % 300 == 0: self._load_faces()

            # YOLO (HAILO)
            if frame_count % 3 == 0:
                self.last_results = self.model(frame, verbose=False, conf=0.35)

            # Draw
            if hasattr(self, 'last_results'):
                for r in self.last_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cls = int(box.cls[0])
                        # Assuming English model names
                        en = self.model.names[cls]
                        tr = TRANSLATION_DICT.get(en, en)
                        cv2.putText(display, f"{tr}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Face (DeepFace CPU)
            if frame_count % 5 == 0: 
                from deepface import DeepFace
                if self.face_cascade:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    temp_faces = []
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        try:
                            # Use explicit model name if needed, assuming CPU allows it
                            embed = DeepFace.represent(frame[y:y+h, x:x+w], model_name='Facenet512', enforce_detection=False)[0]['embedding']
                            match_name, match_score = "Bilinmeyen", -1
                            for idx, enc in enumerate(self.known_encodings):
                                s = self.cosine_similarity(embed, enc)
                                if s > match_score: match_score, match_name = s, self.known_names[idx]
                            if match_score > 0.4:
                                temp_faces.append(match_name)
                                cv2.putText(display, f"{match_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        except: pass
                    current_faces = temp_faces

            cv2.imshow("Hailo Glasses", display)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]: break
            elif key == ord(' '): 
                if hasattr(self, 'last_results'): self.voice.speak(self.describe_scene(frame, self.last_results, current_faces))
            elif key in [ord('r'), ord('R')]: self.read_text(frame)
            elif key in [ord('c'), ord('C')]: self.identify_currency(frame)
            elif key in [ord('x'), ord('X')]: self.identify_color(frame)
            elif key in [ord('f'), ord('F')]: 
                if current_faces: self.voice.speak(f"Bu {', '.join(current_faces)}")
                else: self.voice.speak("Tanıdık kimse yok.")

        self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SmartGlassesAssistant()
    app.run()
