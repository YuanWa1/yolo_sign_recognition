from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

print("Start training: ")
# Train the model
train_results = model.train(
    data = "SignDataset/data.yaml",  # path to dataset YAML
    epochs= 50,  # number of training epochs
    imgsz=640,  # training image size
    batch = 32,
)

print("finished training")
# Evaluate model performance on the validation set
metrics = model.val(save_json = True)

print("testing: ")
results = model.predict(source='./SignDataset/test/images', save=True)

metrics = model.val(data='./SignDataset/data.yaml', split='test', batch = 32, imgsz=640)
