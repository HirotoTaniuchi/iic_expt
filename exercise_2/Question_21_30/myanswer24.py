import cv2
import numpy as np

def gamma_correction(img, c=1, g=2.2):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape


    out = np.zeros( (H, W, C), dtype = np.float )
    out = 255 * ((img / c / 255) ** (1 / g)) #正規化も内部で行う #全画素同時に処理
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori_gamma.jpg").astype(np.uint8) #icv2.mread().astype()
out = gamma_correction(img)

cv2.imwrite("myanswer24img.jpg", out)
cv2.imshow("myanswer24img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
