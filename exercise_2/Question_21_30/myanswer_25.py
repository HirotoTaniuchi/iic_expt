import cv2
import numpy as np

def resize_nearest(img, rate = 1.5):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    print("in: ", img.shape)


    Hout, Wout = int(np.floor(H * rate)), int(np.floor(W * rate))
    out = np.zeros( ( Hout, Wout, C), dtype = np.float) #奇数サイズへの対応
    for y in range(Hout):
        for x in range(Wout):
            for z in range(C):
                out[y, x, z] = img[int(np.floor(y / rate)), int(np.floor(x / rate)), z]
    
    print("out: ",out.shape)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori.jpg").astype(np.uint8) #icv2.mread().astype()
out = resize_nearest(img)

cv2.imwrite("myanswer25img.jpg", out)
cv2.imshow("myanswer25img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 解答例 #参考にしたい点メモ
# def nn_interpolate(img, ax=1, ay=1): #縦横別で比を決められる
# 	H, W, C = img.shape

# 	aH = int(ay * H)
# 	aW = int(ax * W)

# 	y = np.arange(aH).repeat(aW).reshape(aH, -1)
# 	x = np.tile(np.arange(aW), (aH, 1))
# 	y = np.round(y / ay).astype(np.int)
# 	x = np.round(x / ax).astype(np.int) #拡張時に各画素が参照する元の画像のインデックスを並べた配列

# 	out = img[y,x]

# 	out = out.astype(np.uint8)

# 	return out