import os
import shutil
import random
import glob

# --- Configuration ---
SOURCE_DIR = "Turkish Lira"
DEST_DIR = "datasets/currency_cls"
SPLIT_RATIO = 0.8

def prepare_data():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source folder '{SOURCE_DIR}' not found!")
        return

    print(f"Preparing Dataset from '{SOURCE_DIR}'...")
    
    # 1. Setup Directories
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR) # Clean start
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

    # 2. Process each class folder
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_images = 0
    
    for cls in classes:
        print(f"   Processing Class: {cls}")
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        
        # Get all images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            images.extend(glob.glob(os.path.join(src_cls_path, ext)))
            
        random.shuffle(images)
        
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Copy to destinations
        for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
            dest_cls_path = os.path.join(DEST_DIR, split, cls)
            os.makedirs(dest_cls_path, exist_ok=True)
            
            for img_path in imgs:
                try:
                    if os.path.exists(os.path.join(dest_cls_path, os.path.basename(img_path))):
                        os.remove(os.path.join(dest_cls_path, os.path.basename(img_path)))
                    os.link(img_path, os.path.join(dest_cls_path, os.path.basename(img_path)))
                except OSError:
                    # Fallback to copy if hardlink fails
                    shutil.copy(img_path, dest_cls_path)
        
        total_images += len(images)
        print(f"     -> {len(train_imgs)} Train, {len(val_imgs)} Val")

    print(f"\nData Preparation Complete! Total Images: {total_images}")
    print(f"   Output: {DEST_DIR}")

if __name__ == "__main__":
    prepare_data()
