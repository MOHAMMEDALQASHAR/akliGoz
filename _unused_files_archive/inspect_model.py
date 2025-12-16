from ultralytics import YOLO

try:
    model = YOLO("runs/classify/currency_cls/weights/best.pt")
    print("✅ Model loaded successfully.")
    print("📋 Class Names:", model.names)
except Exception as e:
    print(f"❌ Error loading model: {e}")
