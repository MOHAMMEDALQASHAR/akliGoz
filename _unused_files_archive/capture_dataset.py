import cv2
import os
import numpy as np
import time

# --- Configuration ---
DATASET_DIR = "datasets/currency"
CLASSES = ["5", "10", "20", "50", "100", "200"]

class AutoLabeler:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.templates = []
        self._load_templates()
        
        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _load_templates(self):
        template_dir = "currency_templates"
        if not os.path.exists(template_dir):
            print("⚠️ Error: No templates found! Run capture_templates.py first.")
            return

        for f in os.listdir(template_dir):
            if f.endswith(".jpg"):
                path = os.path.join(template_dir, f)
                img = cv2.imread(path, 0)
                if img is None: continue
                kp, des = self.sift.detectAndCompute(img, None)
                if des is not None:
                    val = os.path.splitext(f)[0]
                    self.templates.append((val, kp, des, img.shape[:2])) # Store shape (h, w)
                    print(f"Loaded Template: {val}")

    def get_bounding_box(self, frame_gray, target_class):
        """
        Returns normalized YOLO box (x_center, y_center, w, h) if match found.
        """
        kp_frame, des_frame = self.sift.detectAndCompute(frame_gray, None)
        if des_frame is None or len(des_frame) < 5: return None

        # Find the specific template for the target class
        target_template = next((t for t in self.templates if t[0] == target_class), None)
        if not target_template: return None
        
        val, kp_temp, des_temp, temp_shape = target_template

        matches = self.matcher.knnMatch(des_temp, des_frame, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 12:
            src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Transform template corners to finding frame
                h, w = temp_shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Get bounding rect of the transformed shape
                x, y, w_box, h_box = cv2.boundingRect(dst)
                
                # Ensure box is within frame
                H_frame, W_frame = frame_gray.shape
                x = max(0, x)
                y = max(0, y)
                w_box = min(w_box, W_frame - x)
                h_box = min(h_box, H_frame - y)

                if w_box < 20 or h_box < 20: return None # Too small

                # Normalize for YOLO
                x_center = (x + w_box / 2) / W_frame
                y_center = (y + h_box / 2) / H_frame
                w_norm = w_box / W_frame
                h_norm = h_box / H_frame

                return (x_center, y_center, w_norm, h_norm), dst

        return None, None

def setup_dirs():
    for split in ['train', 'val']:
        os.makedirs(f"{DATASET_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/{split}/labels", exist_ok=True)

def main():
    setup_dirs()
    labeler = AutoLabeler()
    
    if not labeler.templates:
        print("❌ No templates loaded. Please run 'run_capture.sh' first!")
        return

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("\n🎥 AI TRAINING DATA COLLECTOR 🎥")
    print("--------------------------------")
    print("We need to capture ~50 frames for each banknote.")
    print("The system will AUTO-DETECT the bill and save it.")
    print("--------------------------------")

    for cls in CLASSES:
        print(f"\n👉 Prepare to capture: {cls} TL")
        print("   Press 's' to START capturing this note.")
        print("   Press 'n' to SKIP to next note.")
        
        while True:
            ret, frame = cap.read()
            cv2.imshow("Data Collector", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): break
            if key == ord('n'): break
        
        if key == ord('n'): continue

        count = 0
        max_count = 60 # Capture 60 frames per class
        
        while count < max_count:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

            # Try to auto-label
            bbox, quad_pts = labeler.get_bounding_box(gray, cls)
            
            if bbox:
                # Save Data
                timestamp = int(time.time() * 1000)
                # Split 80/20 train/val
                split = "train" if np.random.rand() > 0.2 else "val"
                
                img_name = f"{cls}_{timestamp}.jpg"
                txt_name = f"{cls}_{timestamp}.txt"
                
                # Save Image
                cv2.imwrite(f"{DATASET_DIR}/{split}/images/{img_name}", frame)
                
                # Save Label
                # Class ID is index in CLASSES list
                cls_id = CLASSES.index(cls)
                xc, yc, w, h = bbox
                with open(f"{DATASET_DIR}/{split}/labels/{txt_name}", 'w') as f:
                    f.write(f"{cls_id} {xc} {yc} {w} {h}\n")

                count += 1
                
                # Draw Box
                cv2.polylines(display, [np.int32(quad_pts)], True, (0, 255, 0), 2)
                cv2.putText(display, f"CAPTURING: {count}/{max_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Show {cls} TL clearly...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Data Collector", display)
            if cv2.waitKey(1) == ord('q'): return

    print("\n✅ Data Collection Complete!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
