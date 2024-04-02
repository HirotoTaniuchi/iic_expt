import cv2
import numpy as np

def max_pooling(img, gs = 8): #gs:gridsize
    h, w, c = img.shape
    out = np.empty(( h, w, c ))
    for y in range( h // gs):
        for x in range( w // gs):
            grid = img[ (gs * y) : (gs * (y+1)),  (gs * x) : (gs * (x+1)), :].copy()
            for z in range(3):
                out[ (gs * y) : (gs * (y+1)),  (gs * x) : (gs * (x+1)), z]\
                = np.max(grid[:, :, z])
                #https://deepage.net/features/numpy-max.html#npamax

    return out

img = cv2.imread("imori.jpg")
img2 = max_pooling(img, 8)

cv2.imshow("myanswer8img", img2.astype(np.uint8))
cv2.imwrite("myanswer8img.jpg", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()