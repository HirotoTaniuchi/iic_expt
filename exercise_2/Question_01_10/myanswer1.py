import cv2
import numpy as np

img = cv2.imread("imori.jpg")
img2 = img.copy().astype(np.float32)
print(img.shape)
# print(img.dtype)

#一色だけ取り出す
# red = img[:, :, 2].copy()

H, W, C = img2.shape
img2 = img2[:, :, (2,1,0)]

cv2.imshow("myanswer1img", img2.astype(np.uint8))
cv2.imwrite("myanswer1img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


