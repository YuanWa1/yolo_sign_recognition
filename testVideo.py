from ultralytics import YOLO
import cv2

model = YOLO(r'.\runs\detect\train\weights\best.pt')

video_path = "./SignDataset/TestVideo.MOV"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
    results = model(frame,conf=0.7)
    result = results[0]

    annotated_frame = result.plot()
    cv2.imshow('Hand Sign Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
