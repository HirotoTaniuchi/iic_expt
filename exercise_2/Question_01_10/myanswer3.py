import cv2
import numpy as np


def binarization(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray[ gray < 128 ] = 0
    gray[ gray >= 128 ] = 255
    
    return gray
            


img = cv2.imread("imori.jpg")
img2 = binarization(img)
#img2 = img.copy().astype(np.float32)
# print(img.shape)
# print(img.dtype)

cv2.imshow("myanswer3img", img2.astype(np.uint8))
cv2.imwrite("myanswer3img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()