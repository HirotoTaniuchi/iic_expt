import cv2
import numpy as np



def RGBtoHSV(img):
    "https://algorithm.joho.info/image-processing/opencv-rgb-to-hsv-color-space/"
    h, w, c = img.shape
    out = np.empty((h, w, c))

    for y in range(h):
        for x in range(w):
            b, g, r = img[y, x] / 255.0 #RGBを0~1に正規化
            Max, Min = max(r, g, b), min(r, g, b)#各画素で最大値最初値を取るのはrgbのどれか
            diff = Max - Min

            #H計算
            if Max == Min: h = 0
            elif Max == r: h = 60 * ((g - b) / diff) + 60
            elif Max == g: h = 60 * ((b - r) / diff) + 180
            elif Max == b: h = 60 * ((r - g) / diff) + 300

            #Vの計算
            v = Max

            #Sの計算
            s = Max - Min
            
            out[y][x] = [h, s, v] #この順番で合ってるのか？

    return out 


def HSVtoRGB(img):
    h, w, c = img.shape
    out = np.empty((h, w, c))
    for y in range(h):
        for x in range(w):

            H, S, V = img[y, x]
            C = S
            Hdush = H // 60
            X = C * (1 - abs(Hdush % 2 - 1) )

            if 0 <= Hdush < 1: branch = np.array([C, X, 0])
            elif 1 <= Hdush < 2: branch = np.array([X, C, 0])
            elif 2 <= Hdush < 3: branch = np.array([0, C, X])
            elif 3 <= Hdush < 4: branch = np.array([0, X, C])
            elif 4 <= Hdush < 5: branch = np.array([X, 0, C])
            elif 5 <= Hdush < 6: branch = np.array([C, 0, X])
            else: branch = np.array([0, 0, 0])

            R, G, B = (V - C) * np.array([1, 1, 1]) + branch

            out[y, x] = [B * 255.0, G * 255.0, R * 255.0]

    return out #returnの階層注意





img = cv2.imread("imori.jpg")
hsvimg = RGBtoHSV(img) 
hsvimg[:, :, 0] += 180
img2 = HSVtoRGB (hsvimg)

cv2.imshow("myanswer5img", img2.astype(np.uint8))
cv2.imwrite("myanswer5img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()




