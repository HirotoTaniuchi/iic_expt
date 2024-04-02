import cv2
import numpy as np

def weight(dx, a=-1):
    t = abs(dx)
    if t <= 1:
        return (a+2) * (t**3) - (a+3) * (t**2) + 1
    elif t <=2:
        return a * (t**3) - 5*a * (t**2) + 8*a * t - 4*a
    else:
        return 0

def resize_bicubic(img, rate=1.5): #処理時間長い
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    print("in: ", img.shape)


    # zero padding
    pad = 2
    padded_img = np.zeros((H + pad * 2, W + pad * 2, C), dtype = np.float)
    padded_img[ pad: pad + H, pad: pad + W, :] = img.copy().astype(np.float)


    Hout, Wout = int(np.floor(H * rate)), int(np.floor(W * rate))
    out = np.zeros( ( Hout, Wout, C), dtype = np.float)
    print("out: ", out.shape)
    for y in range(Hout):
        for x in range(Wout):
            for z in range(C):
                relativey, relativex = (y / rate), (x / rate)
                floory, floorx = np.floor(y / rate), np.floor(x / rate)
                dy, dx = (relativey - floory), (relativex - floorx)

                xweights = np.zeros(4, dtype = np.float)
                yweights = np.zeros(4, dtype = np.float)

                for i in range(4): #重み表の作成
                    xweights[i] = weight(dx+1-i)
                    yweights[i] = weight(dy+1-i)

                pixel = 0.
                for j in range(4):
                    for i in range(4):
                        pixel += yweights[j] * xweights[i] * padded_img[int(floory)+1+j, int(floorx)+1+i, z]
                pixel /= np.sum(np.outer(xweights, yweights))
                #https://python.atelierkobato.com/np_outer/ #直積の総和


                out[y, x, z] = pixel


    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    
    return out



img = cv2.imread("imori.jpg")

out = resize_bicubic(img)
cv2.imwrite("myanswer27img.jpg", out)
cv2.imshow("myanwer27img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

