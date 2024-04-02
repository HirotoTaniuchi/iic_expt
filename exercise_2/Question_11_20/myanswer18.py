import cv2
import numpy as np

def BGRtoGray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return gray



def prewitt_filter(img):
    """
    v: vertical, h: horizontal
    """
    #K_size = 3以外だと困る
    K_size = 3
    
    if len(img.shape) == 3:
        img = BGRtoGray(img)
    H, W = img.shape

    # zero padding
    pad = K_size //2
    padded_img = np.zeros((H + pad * 2, W + pad * 2), dtype = np.float)
    padded_img[ pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    # prepare kernel
    # 小数で宣言した方がいい
    K = np.array([ [-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]])


    out = np.zeros( (H, W), dtype = np.float )
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum( K * padded_img[y: y + K_size, x: x + K_size])
    
    
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out



img = cv2.imread("imori.jpg")

out = prewitt_filter(img)
cv2.imwrite("myanswer18img.jpg", out)
cv2.imshow("myanwer18img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()