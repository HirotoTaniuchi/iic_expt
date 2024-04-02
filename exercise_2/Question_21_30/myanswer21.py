import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_normalization(img, a_min=0, b_max=255):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape



    out = np.zeros( (H, W, C), dtype = np.float )
    c_min = np.min(img)
    d_max = np.max(img)

    for z in range(C):
        #c_min = np.min(img[:, :, z]) #こっちで定義すると出力がズレる
        #d_max = np.max(img[:, :, z])

        for y in range(H):
            for x in range(W):
                out[y, x, z] = (img[y, x, z] - c_min) * (b_max - a_min) / (d_max - c_min) + a_min

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("imori_dark.jpg")
out = histogram_normalization(img)

cv2.imwrite("myanswer21img_1.jpg", out)
cv2.imshow("myanswer21img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
plt.hist(out.flatten(), bins=np.arange(256 + 1)) #binsは境界の数なので+1する
plt.show()
fig.savefig("myanswer21img_2.jpg")


# # answer #参考にしたい点メモ
# # c, dはチャネルごとではなく画像全体に対して求めるのが正しい
# 
# def hist_normalization(img, a=0, b=255):
# 	# get max and min
# 	c = img.min()
# 	d = img.max()

# 	out = img.copy()

# 	# normalization
# 	out = (b-a) / (d - c) * (out - c) + a #全画素まとめて処理するときれい
# 	out[out < a] = a
# 	out[out > b] = b
# 	out = out.astype(np.uint8)
	
# 	return out
