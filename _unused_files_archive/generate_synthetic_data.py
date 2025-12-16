import cv2
import os
import numpy as np
import time

# --- Configuration ---
TEMPLATE_DIR = "currency_templates"
DATASET_DIR = "datasets/currency"
CLASSES = ["5", "10", "20", "50", "100", "200"]

def augment_image(image, num_variations=50):
    generated = []
    h, w = image.shape[:2]
    
    for _ in range(num_variations):
        # 1. Random Rotation & Scale
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0))
        
        # 2. Random Brightness
        brightness = np.random.uniform(0.7, 1.3)
        bright = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
        
        # 3. Random Blur
        if np.random.rand() > 0.5:
             ksize = np.random.choice([3, 5])
             bright = cv2.GaussianBlur(bright, (ksize, ksize), 0)
        
        generated.append(bright)
    return generated

def generate_dataset():
    print("🚀 Generating Synthetic Dataset from templates...")
    
    # Setup dirs
    for split in ['train', 'val']:
        os.makedirs(f"{DATASET_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/{split}/labels", exist_ok=True)
        
    for cls in CLASSES:
        img_path = f"{TEMPLATE_DIR}/{cls}.jpg"
        if not os.path.exists(img_path):
            print(f"❌ Missing template: {img_path}")
            continue
            
        print(f"   Processing {cls} TL...")
        original = cv2.imread(img_path)
        if original is None:
            print(f"   ❌ Could not read {img_path}")
            continue
            
        # Resize to standard width if too huge
        if original.shape[1] > 1000:
             scale = 1000 / original.shape[1]
             original = cv2.resize(original, (0,0), fx=scale, fy=scale)

        augmented_images = augment_image(original, num_variations=40) # 40 train
        val_images = augment_image(original, num_variations=10) # 10 val
        
        cls_id = CLASSES.index(cls)
        
        # Save Train
        for i, img in enumerate(augmented_images):
            name = f"{cls}_syn_train_{i}"
            cv2.imwrite(f"{DATASET_DIR}/train/images/{name}.jpg", img)
            # Full image label
            with open(f"{DATASET_DIR}/train/labels/{name}.txt", "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n") # Center, Full width
                
        # Save Val
        for i, img in enumerate(val_images):
            name = f"{cls}_syn_val_{i}"
            cv2.imwrite(f"{DATASET_DIR}/val/images/{name}.jpg", img)
            with open(f"{DATASET_DIR}/val/labels/{name}.txt", "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")
                
    print("✅ Synthetic Dataset Generation Complete!")

if __name__ == "__main__":
    generate_dataset()
