import cv2
import numpy as np



def resize_bilinear(img, rate=1.5):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    print("in: ", img.shape)


    # padding
    padded_img = np.zeros((H + 1, W + 1, C), dtype = np.float)
    padded_img[ 0: H, 0: W] = img.copy().astype(np.float)



    Hout, Wout = int(np.floor(H * rate)), int(np.floor(W * rate))
    out = np.zeros( ( Hout, Wout, C), dtype = np.float)
    print("out: ", out.shape)
    for y in range(Hout):
        for x in range(Wout):
            for z in range(C):
                relativey, relativex = (y / rate), (x / rate)
                floory, floorx = np.floor(y / rate), np.floor(x / rate)
                dy, dx = (relativey - floory), (relativex - floorx)

                for i in range(4):
                    for i in range()

                out[y, x, z] = (1-dy) * (1-dx) * padded_img[int(floory), int(floorx), z]\
                            + (1-dy) * dx * padded_img[int(floory), int(floorx) + 1, z]\
                            + dy * (1-dx) * padded_img[int(floory) + 1, int(floorx), z]\
                            + dy * dx * padded_img[int(floory) + 1, int(floorx) + 1, z]


    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out



img = cv2.imread("imori.jpg")

out = resize_bilinear(img)
cv2.imwrite("myanswer26img.jpg", out)
cv2.imshow("myanwer26img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

