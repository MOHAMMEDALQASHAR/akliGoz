# -*- coding: utf-8 -*-
"""
فحص دقة نموذج التعرف على العملات
Check Currency Model Accuracy
"""

from ultralytics import YOLO
import os

def check_model_info():
    print("=" * 60)
    print("فحص نموذج التعرف على العملات التركية")
    print("Checking Turkish Currency Recognition Model")
    print("=" * 60)
    
    # Check for advanced model
    advanced_model_path = "runs/classify/currency_cls_advanced/weights/best.pt"
    basic_model_path = "runs/classify/currency_cls/weights/best.pt"
    
    model_path = None
    model_type = None
    
    if os.path.exists(advanced_model_path):
        model_path = advanced_model_path
        model_type = "Advanced (50 epochs)"
        print(f"\n✅ Found Advanced Model: {advanced_model_path}")
    elif os.path.exists(basic_model_path):
        model_path = basic_model_path
        model_type = "Basic (10 epochs)"
        print(f"\n✅ Found Basic Model: {basic_model_path}")
    else:
        print("\n❌ No trained model found!")
        print("Please run: python train_classifier_advanced.py")
        return
    
    # Load model
    try:
        print(f"\nLoading model...")
        model = YOLO(model_path)
        
        print("\n" + "=" * 60)
        print("Model Information")
        print("=" * 60)
        print(f"Model Type: {model_type}")
        print(f"Model Path: {model_path}")
        print(f"Model Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Try to get model metrics if available
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)
        
        # Read results.csv if possible
        results_dir = os.path.dirname(os.path.dirname(model_path))
        results_csv = os.path.join(results_dir, "results.csv")
        
        if os.path.exists(results_csv):
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Get header and last line (final epoch)
                    header = lines[0].strip().split(',')
                    last_epoch = lines[-1].strip().split(',')
                    
                    # Find accuracy columns
                    for i, col in enumerate(header):
                        col = col.strip()
                        if 'top1' in col.lower() or 'accuracy' in col.lower():
                            try:
                                value = float(last_epoch[i])
                                print(f"{col}: {value * 100:.2f}%")
                            except:
                                pass
                        elif 'loss' in col.lower():
                            try:
                                value = float(last_epoch[i])
                                print(f"{col}: {value:.4f}")
                            except:
                                pass
        
        # Get class names
        if hasattr(model, 'names'):
            print("\n" + "=" * 60)
            print("Supported Currency Classes")
            print("=" * 60)
            names = model.names
            if isinstance(names, dict):
                for idx, name in sorted(names.items()):
                    print(f"  {idx}: {name} TL")
            print(f"\nTotal Classes: {len(names)}")
        
        # Validate on test data
        dataset_path = 'c:/Users/ta775/OneDrive/Desktop/akilligoz-main/datasets/currency_cls'
        val_path = os.path.join(dataset_path, 'val')
        
        if os.path.exists(val_path):
            print("\n" + "=" * 60)
            print("Running Validation on Test Set...")
            print("=" * 60)
            
            try:
                metrics = model.val(data=dataset_path, split='val')
                
                print(f"\n🎯 Validation Accuracy: {metrics.top1 * 100:.2f}%")
                print(f"📊 Top-5 Accuracy: {metrics.top5 * 100:.2f}%")
                
                print("\n" + "=" * 60)
                print("Expected Performance")
                print("=" * 60)
                if model_type.startswith("Advanced"):
                    print("النموذج المتقدم - Advanced Model:")
                    print("  • Expected Accuracy: 90-95%+")
                    print("  • Training: 50 epochs")
                    print("  • Image Size: 320px")
                    print("  • Data Augmentation: Advanced")
                else:
                    print("النموذج الأساسي - Basic Model:")
                    print("  • Expected Accuracy: 75-85%")
                    print("  • Training: 10 epochs")
                    print("  • Image Size: 224px")
                
            except Exception as e:
                print(f"Could not run validation: {e}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")

if __name__ == "__main__":
    check_model_info()
