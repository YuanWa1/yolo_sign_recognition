from ultralytics import YOLO
import cv2

model = YOLO(r'.\runs\detect\train5\weights\best.pt')

image = cv2.imread('./SignDataset/testImage2.png')

results = model(image)
result = results[0]
resultImage = result.plot()
cv2.imshow('result', resultImage)

cv2.imwrite('test1.jpg', resultImage )


cv2.waitKey(0)