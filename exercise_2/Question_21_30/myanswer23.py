import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(img, Zmax = 255):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    S = H * W * C


    hist, _ = np.histogram(img.flatten(), bins = np.arange(0, 257), range = (0, 256))
    # https://deepage.net/features/numpy-histogram.html
    #histの指定方法, 末端に注意
    sums = np.zeros(256, dtype = np.uint64) #度数の順次和を記憶しておく=毎度全探索しなくてよいので計算早い
    #ここuint64 #度数は256を余裕で超えるのでオーバーフロー注意
    sums[0] = hist[0]
    for i in range(255):
        sums[i+1] = sums[i] + hist[i+1]




    out = np.zeros( (H, W, C), dtype = np.float )
    for y in range(H):
        for x in range(W):
            for z in range(C):
                # print("ppp:", img[y,x,z])
                out[y, x, z] = Zmax * (sums[img[y, x, z]] / S)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori_dark.jpg").astype(np.uint8) #icv2.mread().astype()
out = histogram_equalization(img)

cv2.imwrite("myanswer23img_1.jpg", out)
cv2.imshow("myanswer23img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
plt.hist(out.flatten(), bins=255, rwidth=0.8, range=(0, 255)) #解答の条件
plt.show()
fig.savefig("myanswer23img_2.jpg")