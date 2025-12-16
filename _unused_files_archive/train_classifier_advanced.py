# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

def train_classifier_advanced():
    """
    Advanced training script for Turkish Currency Classification
    
    Improvements:
    1. More epochs for better learning
    2. Larger image size for more details
    3. Data augmentation enabled
    4. Better learning rate scheduling
    5. Optimized hyperparameters
    """
    
    print("="*60)
    print("Advanced Turkish Currency Classifier Training")
    print("="*60)
    
    # Check if dataset exists
    dataset_path = 'c:/Users/ta775/OneDrive/Desktop/akilligoz-main/datasets/currency_cls'
    if not os.path.exists(dataset_path):
        print("ERROR: Dataset not found!")
        print("Please run 'prepare_classification_data.py' first")
        return
    
    # Count training samples
    train_path = os.path.join(dataset_path, 'train')
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print(f"\nFound {len(classes)} currency classes:")
    for cls in sorted(classes):
        cls_path = os.path.join(train_path, cls)
        count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  - {cls} TL: {count} images")
    
    # 1. Load Pretrained Classification Model
    # Using YOLOv8n-cls (nano) - fast and accurate for classification
    model = YOLO('yolov8n-cls.pt')
    
    print("\n" + "="*60)
    print("Starting Training with Advanced Settings...")
    print("="*60)
    
    # 2. Train with Advanced Settings
    results = model.train(
        data=dataset_path,
        
        # Training Duration
        epochs=50,              # Increased from 10 to 50 for better learning
        patience=10,            # Early stopping if no improvement for 10 epochs
        
        # Image Settings
        imgsz=320,              # Increased from 224 to 320 for more detail
        
        # Batch and Performance
        batch=16,               # Batch size (adjust based on GPU memory)
        workers=4,              # Number of data loading workers
        
        # Data Augmentation (helps model generalize better)
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation  
        hsv_v=0.4,             # Value/brightness augmentation
        degrees=15,            # Rotation augmentation (+/- 15 degrees)
        translate=0.1,         # Translation augmentation
        scale=0.5,             # Scale augmentation
        shear=0.0,             # Shear augmentation
        perspective=0.0,       # Perspective augmentation
        flipud=0.5,            # Vertical flip probability
        fliplr=0.5,            # Horizontal flip probability
        mosaic=1.0,            # Mosaic augmentation probability
        mixup=0.1,             # Mixup augmentation probability
        
        # Optimizer Settings
        optimizer='AdamW',     # AdamW optimizer (better than SGD for classification)
        lr0=0.001,             # Initial learning rate
        lrf=0.01,              # Final learning rate (lr0 * lrf)
        momentum=0.937,        # Momentum
        weight_decay=0.0005,   # Weight decay for regularization
        warmup_epochs=3.0,     # Warmup epochs
        warmup_momentum=0.8,   # Warmup momentum
        
        # Other Settings
        cos_lr=True,           # Use cosine learning rate scheduler
        label_smoothing=0.1,   # Label smoothing (prevents overconfidence)
        dropout=0.2,           # Dropout for regularization
        
        # Output Settings
        name='currency_cls_advanced',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        save=True,
        save_period=10,        # Save checkpoint every 10 epochs
        
        # Validation
        val=True,
        plots=True             # Generate training plots
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: runs/classify/currency_cls_advanced/weights/best.pt")
    print(f"Last model saved to: runs/classify/currency_cls_advanced/weights/last.pt")
    print("\nTo use this model, update the path in main_glasses.py:")
    print("  custom_model_path = 'runs/classify/currency_cls_advanced/weights/best.pt'")
    print("="*60)

if __name__ == "__main__":
    train_classifier_advanced()
