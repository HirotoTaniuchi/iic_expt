import cv2
import numpy as np



def affin_transform(img, affin, xrate = 1.0, yrate = 1.0): #rate[0]はx方向の倍率, rate[1]はy方向の倍率
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    print("in: ", img.shape)

    #出力画像
    outH, outW = int(yrate * H), int(xrate * W)
    out = np.zeros((outH, outW, C))

    #アフィン変換の逆行列
    inv_affin = np.linalg.inv(affin)

    #最近傍法で画像を変換
    x,y = np.mgrid[:outW,:outH]
    xy_after = np.dstack((x, y, np.ones((outW, outH)))) #アフィン変換後の各ピクセルの座標。変換のためにベクトルの末尾に1をつける(outH x outW x 3)
    ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[:, :, :2] #変換後の各ピクセルが参照すべき変換前の座標を計算(outH x outW x 2)

    for y in range(outH):
        for x in range(outW):
            ref_y, ref_x = int(ref_xy[x, y, 1]), int(ref_xy[x, y, 0]) #最近傍の参照先
            if (ref_y < 0) or (ref_y > H-1) or (ref_x < 0) or (ref_x > W-1): #参照先がなかったら0で埋める
                out[y, x, :] = 0
            else:
                out[y, x, :] = img[ref_y, ref_x, :]

    print("out: ", out.shape)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)



    return out



img = cv2.imread("imori.jpg")

affin1 = np.array([[1.3, 0, 0], [0, 0.8, 0], [0, 0, 1.]]) #拡大率から, a = 1.3, d = 0.8
affin2 = np.array([[1, 0, 30], [0, 1, -30], [0, 0, 1.]]) #移動量から, tx = 30, ty = -30
out = affin_transform(img, affin1, xrate = 1.3, yrate = 0.8)

cv2.imwrite("myanswer29img_1.jpg", out)
cv2.imshow("myanwer29img_1", out)
cv2.waitKey(0)
cv2.destroyWindow("out")


out = affin_transform(img, np.matmul(affin2, affin1), xrate = 1.3, yrate = 0.8)
cv2.imwrite("myanswer29img_2.jpg", out)
cv2.imshow("myanwer29img_2", out)
cv2.waitKey(0)
cv2.destroyWindow("out")

cv2.destroyAllWindows()
