import cv2
import numpy as np


def BGRtoGray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return gray



def sobel_filter(img, direction = "v"):
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
    if direction == "v":
        K = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ])
    else:
        K = np.array([ [1, 0, -1], [2, 0, -2], [1, 0, -1] ])


    out = np.zeros( (H, W), dtype = np.float )
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum( K * padded_img[y: y + K_size, x: x + K_size])
    
    
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out



img = cv2.imread("imori.jpg")


out_v = sobel_filter(img, direction = "v")
cv2.imwrite("myanswer15img_v.jpg", out_v)
cv2.imshow("myanwer15img_v", out_v)
cv2.waitKey(0)
cv2.destroyWindow('result_v')


out_h = sobel_filter(img, direction = "h")
cv2.imwrite("myanswer15img_h.jpg", out_h)
cv2.imshow("myanwer15img_h", out_h)
cv2.waitKey(0)
cv2.destroyWindow('result_v')

cv2.destroyAllWindows()
