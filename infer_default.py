from ultralytics import YOLO

# Load a model
model = YOLO("C:/Users/kadir/dev/ultralytics/yolov8-glasses-sunglasses-bg.pt")  # load a pretrained model (recommended for training)

# Use the model
model(source=0,show=True)