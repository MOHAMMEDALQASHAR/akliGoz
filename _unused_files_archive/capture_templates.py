import cv2
import os
import time

def capture_templates():
    save_dir = "currency_templates"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("\n💰 CURRENCY TEMPLATE CAPTURE TOOL 💰")
    print("---------------------------------------")
    print("Press the following keys to save a template:")
    print("  '2' -> 200 TL")
    print("  '1' -> 100 TL")
    print("  '5' -> 50 TL")
    print("  '0' -> 20 TL (Think 2-zero)")
    print("  '9' -> 10 TL")
    print("  'f' -> 5 TL")
    print("\n  'q' -> Quit")
    print("---------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        
        # Guide Box
        h, w = frame.shape[:2]
        cv2.rectangle(display, (int(w*0.15), int(h*0.2)), (int(w*0.85), int(h*0.8)), (0, 255, 0), 2)
        cv2.putText(display, "Place Note Inside & Press Key", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Templates", display)

        key = cv2.waitKey(1) & 0xFF

        captured_value = None
        if key == ord('2'): captured_value = "200"
        elif key == ord('1'): captured_value = "100"
        elif key == ord('5'): captured_value = "50"
        elif key == ord('0'): captured_value = "20"
        elif key == ord('9'): captured_value = "10"
        elif key == ord('f'): captured_value = "5"
        elif key == ord('q'): break

        if captured_value:
            filename = os.path.join(save_dir, f"{captured_value}.jpg")
            # Save the raw frame region or full frame? Better to crop to box to reduce noise.
            # But user might not align perfectly. Let's save full frame but user should fill screen.
            # Actually, let's crop to the green box to enforce quality.
            
            roi = frame[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)]
            cv2.imwrite(filename, roi)
            print(f"✅ Saved {captured_value} TL to {filename}")
            
            # Flash effect
            cv2.rectangle(display, (0,0), (w,h), (255,255,255), -1)
            cv2.imshow("Capture Templates", display)
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_templates()
