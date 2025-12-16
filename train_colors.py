from ultralytics import YOLO

def train():
    # 1. Load the Nano Classification Model (Pretrained)
    # This is the lightest model, perfect for Raspberry Pi
    model = YOLO('yolov8n-cls.pt') 

    print("🚀 Starting Training on Local Dataset...")
    
    # 2. Train the model
    # data: Path to the folder containing 'train' and 'val' folders
    # epochs: 20 is usually enough for colors (simple features)
    # imgsz: 224 is standard for classification
    results = model.train(
        data='datasets/colors', 
        epochs=30, 
        imgsz=224,
        project='runs/classify',
        name='color_model_custom',
        device='cpu' # CHANGED TO CPU to avoid CUDA errors
    )

    print("\n✅ Training Complete!")
    print(f"   Best Model Saved at: {results.save_dir}/weights/best.pt")
    print("   You can now copy this 'best.pt' to your Raspberry Pi.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Make sure your 'datasets/colors' folder exists and has 'train' and 'val' subfolders inside.")
