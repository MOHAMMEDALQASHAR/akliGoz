# 🧠 How to Train on Google Colab
If your computer is slow, you can train your massive dataset on **Google Colab** (Free GPU).

## Step 1: Zip Your Data
1.  Go to your `SmartGlasses_Project` folder.
2.  Right-click the `my_currency_data` folder.
3.  Select **Compress** or **Zip** to create `my_currency_data.zip`.

## Step 2: Open Colab
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **New Notebook**.
3.  Go to **Runtime > Change runtime type** and select **T4 GPU**.

## Step 3: Upload Data
1.  Click the **Folder Icon** (Files) on the left sidebar.
2.  Drag and drop your `my_currency_data.zip` there.

## Step 4: The Training "Script"
Copy and paste this code into a cell and run it (Shift + Enter):

```python
# 1. Install YOLO
!pip install ultralytics

# 2. Unzip Dataset
!unzip -q my_currency_data.zip -d ./raw_data

# 3. Prepare Data (Split Train/Val)
import os
import shutil
import random
import glob

# Setup
SOURCE_DIR = "./raw_data"
DEST_DIR = "./datasets/currency_cls"

# Create proper folder structure for YOLO Classification
for split in ['train', 'val']:
    os.makedirs(f"{DEST_DIR}/{split}", exist_ok=True)

class_names = os.listdir(SOURCE_DIR)
print(f"Found classes: {class_names}")

for cls in class_names:
    src_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(src_path): continue
    
    # Get all images
    images = glob.glob(f"{src_path}/*.*")
    random.shuffle(images)
    
    # Split 80/20
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    # Copy
    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        os.makedirs(f"{DEST_DIR}/{split}/{cls}", exist_ok=True)
        for img in imgs:
            shutil.copy(img, f"{DEST_DIR}/{split}/{cls}/")

print("✅ Data Organized!")

# 4. Train the Model
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt') # Load generic model

results = model.train(
    data='./datasets/currency_cls',
    epochs=20,     # Train longer since Colab is fast
    imgsz=224,
    batch=64
)

print("🎉 Training Done!")
```

## Step 5: Download the Brain
1.  After training finishes, go to the Files sidebar on the left.
2.  Navigate to `runs` -> `classify` -> `train` -> `weights`.
3.  Right-click `best.pt` and **Download**.
4.  Rename it to `my_currency_model.pt` and put it on your computer.

That's it! You have trained a professional AI model.
