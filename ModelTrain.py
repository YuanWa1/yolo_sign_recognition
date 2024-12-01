from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library

# Load a pre-trained YOLO model
model = YOLO("yolo11n.pt")  # Load the YOLO model from a given .pt file

print("Start training: ")

# Start training the model with the specified parameters
train_results = model.train(
    data="SignDataset/data.yaml",  # Path to the dataset YAML file which contains data and class info
    epochs=50,  # Number of training epochs (iterations over the full dataset)
    imgsz=640,  # The size of the images for training (resize all images to 640x640)
    batch=32,   # The batch size (number of images processed per step)
)

print("Finished training")

# Evaluate the model performance on the validation set
metrics = model.val(save_json=True)  # Run validation and save the results as a JSON file

print("Testing: ")

# Run inference (prediction) on the test images and save the results
results = model.predict(source='./SignDataset/test/images', save=True)  # Test the model on the images in the test set

# Evaluate model performance on the test split of the dataset (using 'test' data)
metrics = model.val(data='./SignDataset/data.yaml', split='test', batch=32, imgsz=640)

