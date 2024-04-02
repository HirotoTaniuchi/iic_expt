import cv2
import numpy as np

def BGRtoGray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # Gray scale
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return gray

def diffrential_filter(img, K_size = 3, direction = "v"):
    """
    v: vertical, h: horizontal
    """
    
    if len(img.shape) == 3:
        img = BGRtoGray(img)
    H, W = img.shape
    

    # zero padding
    pad = K_size //2
    padded_img = np.zeros((H + pad * 2, W + pad * 2), dtype = np.float)
    padded_img[ pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    # prepare kernel
    K = np.zeros((K_size, K_size))
    K[pad, pad] = 1 #中心(に最も近い画素)を1に
    if direction == "v":
        K[pad - 1, pad] = -1 #上に隣接する1画素を-1に
    else:
        K[pad, pad - 1] = -1 #左に隣接する1画素を-1に

    print("kernel: \n",K)

    out = np.zeros( (H, W), dtype = np.float )
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum( K * padded_img[y: y + K_size, x: x + K_size])
    
    #np.clipしないと変なの出力される
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out


img = cv2.imread("imori.jpg")

out = diffrential_filter(img, K_size = 3, direction = "v")
cv2.imshow("myanswer14img_v", out)
cv2.imwrite("myanswer14img_v.jpg", out)

out = diffrential_filter(img, K_size = 3, direction = "h")
cv2.imwrite("myanswer14img_h.jpg", out)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
# answer (2枚出力する方法)
# Save result
cv2.imwrite("out_v.jpg", out_v)
cv2.imshow("result_v", out_v)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('result_v',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('result_v')

cv2.imwrite("out_h.jpg", out_h)
cv2.imshow("result_h", out_h)
# loop if not get ESC or click x
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty('result_h',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('result_h')
cv2.destroyAllWindows()
"""