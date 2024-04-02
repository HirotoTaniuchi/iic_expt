import cv2
import numpy as np


# def BGRtoY(img):

#     Y = np.zeros((img.shape[0], img.shape[1]))
#     weight = [0.2126, 0.7152, 0.0722] 
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             for k in range(3):
#                 Y[i,j] += img[i, j, k] * weight[k]
#     return Y


def BGRtoY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    print(out.shape)
    print(type(out))
    out = out.astype(np.uint8)
    return out





img = cv2.imread("imori.jpg")
img2 = BGRtoY(img)
#img2 = img.copy().astype(np.float32)
# print(img.shape)
# print(img.dtype)

cv2.imshow("myanswer2img", img2.astype(np.uint8))
cv2.imwrite("myanswer2img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()



