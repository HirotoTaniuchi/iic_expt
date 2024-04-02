import cv2
import numpy as np

def color_reduction(img):
    img[img < 64] = 32
    img[(64 <= img) & (img < 128)] = 96
    img[(128 <= img) & (img < 192)] = 160
    img[(192 <= img) & (img < 256)] = 224
    #h, w, c = img.shape
    #out = np.empty((h, w, c))

    return img

img = cv2.imread("imori.jpg")
img2 = color_reduction (img)

cv2.imshow("myanswer6img", img2.astype(np.uint8))
cv2.imwrite("myanswer6img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()



