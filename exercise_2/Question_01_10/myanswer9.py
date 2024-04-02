import cv2
import numpy as np


def zero_pudding(img, n_insert=1):
    """
    外枠を全てゼロで追加する関数
    モノクロ画像用(チャネルは1つ)
    n_insertは縁の幅ピクセル
    https://www.higashisalary.com/entry/opencv-add-frame
    """
    #枠追加処理(上下)
    bk1=np.zeros((n_insert,img.shape[1]),np.uint8)
    array=np.insert(img, 0, bk1, axis=0)
    array=np.insert(array, array.shape[0], bk1, axis=0)
    #枠追加処理(左右)
    bk2=np.zeros((array.shape[0],n_insert),np.uint8)
    array=np.insert(array, [0], bk2, axis=1)
    array=np.insert(array, [array.shape[1]], bk2, axis=1)
    return array

def gaussian_filter(img):

    H, W, C = img.shape[0], img.shape[1], img.shape[2]
    out = np.empty((H, W, C))

    #np.arrayはndarrayを作成する関数
    K =  (1/16) * np.array( [ [1, 2, 1], [2, 4, 2], [1, 2, 1] ] )

    for z in range(C):
        #チャネルごとにパディングする
        padded_img = zero_pudding(img[:, :, z], 1)
        for y in range(H):
            for x in range(W):
                #要素積はアスタリスク
                out[y, x, z] = np.sum( K * padded_img[ y:(y+3), x:(x+3) ])

    return out


img = cv2.imread("imori_noise.jpg")
img2 = gaussian_filter(img)

cv2.imshow("myanswer9img", img2.astype(np.uint8))
cv2.imwrite("myanswer9img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


#answer　#参考にしたい点メモ
"""
# Gaussian filter
# K_size を偶数にするとフィルタの中心がズレるので、左右非対称な処理になる
# 4x4ならこんなカーネルになる
# OOOO
# OOOO
# OOXO
# OOOO


def gaussian_filter(img, K_size=3, sigma=1.3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		img = np.expand_dims(img, axis=-1)  #モノクロでもチャネル次元を作る
		H, W, C = img.shape

		
	## Zero padding
    #拡大したゼロ配列作ってから代入する = めっちゃ楽
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float) #タイプ宣言大事
	out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

	## prepare Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad + K_size):
		for y in range(-pad, -pad + K_size):
			K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
	K /= (2 * np.pi * sigma * sigma)
	K /= K.sum()

	tmp = out.copy()

	# filtering
	for y in range(H):
		for x in range(W):
			for c in range(C):
				out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    
    #最大値最小値を決定
    #https://numpy.org/doc/stable/reference/generated/numpy.clip.html
	out = np.clip(out, 0, 255)
    #ここで削る↓
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out
"""