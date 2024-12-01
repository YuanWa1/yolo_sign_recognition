import cv2  # Import the OpenCV library for image processing

# Read the image from the specified file path
image = cv2.imread('SignDataset/train/images/Z29_jpg.rf.e4a03cf2abb6d36478b465afe1ec7421.jpg')

# Apply Gaussian blur to the image with a kernel size of (5, 5) and no specific sigma for X
blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

# Display the original image in a window titled 'Original Image'
cv2.imshow('Original Image', image)

# Display the blurred image in a window titled 'Blurred Image'
cv2.imshow('Blurred Image', blurred_image)

# Save the blurred image to the specified file path
cv2.imwrite('SignDataset/train/images/Blur_Z29_jpg.rf.e4a03cf2abb6d36478b465afe1ec7421.jpg', blurred_image)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

