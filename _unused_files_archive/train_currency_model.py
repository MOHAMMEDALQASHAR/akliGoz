from ultralytics import YOLO
import os
import yaml

def train_model():
    # 1. Create data.yaml
    dataset_path = os.path.abspath("datasets/currency")
    
    data_yaml = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: '5 TL',
            1: '10 TL',
            2: '20 TL',
            3: '50 TL',
            4: '100 TL',
            5: '200 TL'
        }
    }
    
    with open(f"{dataset_path}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)

    print("🚀 Starting YOLOv8 Training...")
    print(f"   Dataset: {dataset_path}")
    
    # 2. Load Model
    model = YOLO('yolov8n.pt') 

    # 3. Train
    # 50 epochs is usually enough for such a small, specific dataset
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name='currency_yolo'
    )
    
    print("✅ Training Complete!")
    print(f"   Best Model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_model()
