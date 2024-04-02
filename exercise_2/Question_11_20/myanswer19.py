import cv2
import numpy as np

def BGRtoGray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return gray



def LoG_filter(img, K_size=5, sigma=3.0):
    if len(img.shape) == 3:
        img = BGRtoGray(img)
    H, W = img.shape


    # zero padding
    pad = K_size //2
    padded_img = np.zeros((H + pad * 2, W + pad * 2), dtype = np.float)
    padded_img[ pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum() #平均はとる
    #print(K)


    out = np.zeros( (H, W), dtype = np.float )
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum( K * padded_img[y: y + K_size, x: x + K_size])
    
    
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out


img = cv2.imread("imori_noise.jpg")

out = LoG_filter(img)
cv2.imwrite("myanswer19img.jpg", out)
cv2.imshow("myanwer19img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
