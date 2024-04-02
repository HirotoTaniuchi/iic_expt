import cv2
import numpy as np

def median_filter(img, K_size=3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape


    # zero padding
    pad = K_size //2
    padded_img = np.zeros((H + pad * 2, W + pad * 2, C), dtype = np.float)
    padded_img[ pad: pad + H, pad: pad + W] = img.copy().astype(np.float)


    out = np.zeros( (H, W, C), dtype = np.float )
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[y, x, c] = np.median( padded_img[y: y + K_size, x: x + K_size, c])


    #クリップはなぜ必要なのか。なくても動きはする
    #out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out



img = cv2.imread("imori_noise.jpg")
out = median_filter(img, K_size = 3)

cv2.imshow("myanswer10img", out)
cv2.imwrite("myanswer10img.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()



