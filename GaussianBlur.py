import cv2

image = cv2.imread('SignDataset/train/images/Z29_jpg.rf.e4a03cf2abb6d36478b465afe1ec7421.jpg')

blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)

cv2.imwrite('SignDataset/train/images/Blur_Z29_jpg.rf.e4a03cf2abb6d36478b465afe1ec7421.jpg', blurred_image)

# 等待键盘输入
cv2.waitKey(0)
cv2.destroyAllWindows()
