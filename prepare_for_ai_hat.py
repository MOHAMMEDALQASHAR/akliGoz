from ultralytics import YOLO
import os

def export_models():
    print("🚀 Starting Model Export to ONNX...")
    print("   (This format is required before converting to AI Hat/Hailo)")

    # 1. Export Object Detection Model (YOLOv8n)
    print("\n📦 Exporting YOLOv8n (Objects)...")
    try:
        model = YOLO('yolov8n.pt')
        # Export to ONNX with dynamic axes usually, but Hailo prefers static usually 640x640
        path = model.export(format='onnx', imgsz=640, opset=12) 
        print(f"✅ Export Success: {path}")
    except Exception as e:
        print(f"❌ Error exporting YOLOv8n: {e}")

    # 2. Export Currency Model (Custom)
    print("\n💰 Exporting Currency Model...")
    currency_path = "runs/classify/currency_cls_advanced/weights/best.pt"
    if os.path.exists(currency_path):
        try:
            model_curr = YOLO(currency_path)
            # Classification usually 224x224
            path_curr = model_curr.export(format='onnx', imgsz=224, opset=12)
            print(f"✅ Export Success: {path_curr}")
        except Exception as e:
            print(f"❌ Error exporting Currency Model: {e}")
    else:
        print(f"⚠️ Custom currency model not found at {currency_path}")

    # 3. Export Color Model (Custom)
    print("\n🎨 Exporting Color Model...")
    color_path = "runs/classify/color_model_custom/weights/best.pt"
    if os.path.exists(color_path):
        try:
            model_color = YOLO(color_path)
            # Classification usually 224x224
            path_color = model_color.export(format='onnx', imgsz=224, opset=12)
            print(f"✅ Export Success: {path_color}")
        except Exception as e:
            print(f"❌ Error exporting Color Model: {e}")
    else:
        print(f"⚠️ Custom color model not found at {color_path}")

if __name__ == '__main__':
    export_models()
