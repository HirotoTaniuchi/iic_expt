import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_transformation(img, m0=128, s0=52):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape



    out = np.zeros( (H, W, C), dtype = np.float )
    m = np.mean(img)
    s = np.std(img)

    
    out = (s0 / s) * (img - m) + m0
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori_dark.jpg")
out = histogram_transformation(img)

cv2.imwrite("myanswer22img_1.jpg", out)
cv2.imshow("myanswer22img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
plt.hist(out.flatten(), bins=np.arange(256 + 1))
plt.show()
fig.savefig("myanswer22img_2.jpg")