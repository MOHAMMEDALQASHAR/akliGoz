from ultralytics import YOLO

def train_classifier():
    # 1. Load Pretrained Classification Model
    model = YOLO('yolov8n-cls.pt') 

    # 2. Train
    model.train(
        data='c:/Users/ta775/OneDrive/Desktop/akilligoz-main/datasets/currency_cls',
        epochs=10, 
        imgsz=224, 
        name='currency_cls'
    )

if __name__ == "__main__":
    train_classifier()
