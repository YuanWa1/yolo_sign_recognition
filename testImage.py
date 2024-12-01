from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library
import cv2  # Import the OpenCV library for image processing

# Load the trained YOLO model from the best.pt file
model = YOLO(r'.\runs\detect\train5\weights\best.pt')  # Path to the best trained model weights

# Read the image from the file path using OpenCV
image = cv2.imread('./SignDataset/testImage2.png')  # Path to the test image

# Run the YOLO model to get predictions on the input image
results = model(image)  # Inference on the image

# Extract the first result (assumes batch size of 1)
result = results[0]

# Plot the results (bounding boxes, labels, etc.) on the image
resultImage = result.plot()  # Plot the detection results on the image

# Display the resulting image with detections in a window
cv2.imshow('Result', resultImage)

# Save the resulting image to a file (with detections marked)
cv2.imwrite('test1.jpg', resultImage)  # Save the result image with detections

# Wait indefinitely until a key is pressed to close the window
cv2.waitKey(0)
