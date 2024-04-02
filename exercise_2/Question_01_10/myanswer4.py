import cv2
import numpy as np

def otsu_binarization(img):
    gray = 0.2126 * img[:, :, 2].copy() + 0.7152 * img[:, :, 1].copy() + 0.0722 * img[:, :, 0].copy()
    #print(gray.shape)
    #print(type(gray))
    gray = gray.astype(np.uint8)
    row, column = gray.shape[0], gray.shape[1]

    
    #thresh, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)＃閾値関数使えば早い
    #https://qiita.com/ToppaD/items/c0bd354bc7dfcc4318a4
    
    
    thresh, maxSb2= 0, 0

    for i in range(256):

        #https://note.nkmk.me/python-numpy-count/#ndarray-npcount_nonzero
        w0 = np.count_nonzero(gray < i) / row * column #H,Wの方がかっこいいかも
        w1 = np.count_nonzero(gray >= i) / row * column
        #https://kino-code.com/python-numpy-mean/
        M0 = np.mean(gray [gray < i])
        M1 = np.mean(gray [gray >= i])

        if w0 * w1 * ((M0 - M1)**2) > maxSb2:
            maxSb2 = w0 * w1 * ((M0 - M1)**2)
            thresh = i


    print("thresh: ", thresh)
    gray[gray < thresh] = 0
    gray[gray >= thresh ] = 255

    return gray





img = cv2.imread("imori.jpg")
img2 = otsu_binarization(img)
#img2 = img.copy().astype(np.float32)
# print(img.shape)
# print(img.dtype)

cv2.imshow("myanswer4img", img2.astype(np.uint8))
cv2.imwrite("myanswer4img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


#answer
# Otsu Binalization
# def otsu_binarization(img):
# 	max_sigma = 0
# 	max_t = 0
# 	H, W = img.shape
# 	# determine threshold
# 	for _t in range(1, 256):
# 		v0 = out[np.where(out < _t)]
# 		m0 = np.mean(v0) if len(v0) > 0 else 0.
# 		w0 = len(v0) / (H * W)
# 		v1 = out[np.where(out >= _t)]
# 		m1 = np.mean(v1) if len(v1) > 0 else 0.
# 		w1 = len(v1) / (H * W)
# 		sigma = w0 * w1 * ((m0 - m1) ** 2)
# 		if sigma > max_sigma:
# 			max_sigma = sigma
# 			max_t = _t

# 	# Binarization
# 	print("threshold >>", max_t)
# 	th = max_t
# 	out[out < th] = 0
# 	out[out >= th] = 255

# 	return out