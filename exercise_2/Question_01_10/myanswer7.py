import cv2
import numpy as np

def mean_pooling(img, gridsize):
    h, w, c = img.shape
    out = np.empty(( h, w, c ))
    for y in range( h // gridsize):
        for x in range( w // gridsize):
            grid = img[ (gridsize * y) : (gridsize * (y + 1)),  (gridsize * x) : (gridsize * (x + 1)), :].copy()
            for z in range(3):
                out[ (gridsize * y) : (gridsize * (y + 1)),  (gridsize * x) : (gridsize * (x + 1)), z]\
                = np.average(grid[:, :, z])
                #https://deepage.net/features/numpy-average.html
                #(h,w,c) -> (1,1,c)で平均を出してくれる関数はないのか？？あったら便利

    return out




img = cv2.imread("imori.jpg")
img2 = mean_pooling(img,8)

cv2.imshow("myanswer7img", img2.astype(np.uint8))
cv2.imwrite("myanswer7img.jpg", img2.astype(np.uint8))#出力する時点で整数型に変えてあったほうがいいかも
cv2.waitKey(0)
cv2.destroyAllWindows()


#answer #参考にしたい点メモ
# def average_pooling(img, G=8):
#     out = img.copy()

#     H, W, C = img.shape
#     Nh = int(H / G)
#     Nw = int(W / G)

#     for y in range(Nh):
#         for x in range(Nw):
#             for c in range(C):
#                 out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)
#                   #この時点で整数型に変えてあるので出力が画像表示に適している    
#     return out