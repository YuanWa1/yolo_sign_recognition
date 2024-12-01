from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library
import cv2  # Import OpenCV library for video processing and display

# Load the trained YOLO model from the best.pt file
model = YOLO(r'.\runs\detect\train\weights\best.pt')  # Path to the trained YOLO model weights

# Path to the input video file
video_path = "./SignDataset/TestVideo.MOV"
cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV

# Process each frame from the video until the video ends
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If the frame could not be read, exit the loop
        break

    # Resize the frame to reduce processing time (scaling by 50% in both axes)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Run the YOLO model to detect objects in the frame with a confidence threshold of 0.7
    results = model(frame, conf=0.7)

    # Extract the first result (assuming only one frame is processed at a time)
    result = results[0]

    # Annotate the frame with detection results (bounding boxes, labels, etc.)
    annotated_frame = result.plot()

    # Display the annotated frame in a window titled 'Hand Sign Detection'
    cv2.imshow('Hand Sign Detection', annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
