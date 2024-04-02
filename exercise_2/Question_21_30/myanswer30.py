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

def rotate_matrix1(theta = 0.):
    """
    thetaは度数法
    thetaだけ反時計回りに回転する行列を返す
    """
    rad = 2 * np.pi * (theta / 360) #弧度法に直す
    cos = np.cos(rad)
    sin = np.sin(rad)

    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

def rotate_matrix2(img, theta = 0.):
    """
    画像の中心を回転軸にするようにrotate_matrix1を拡張する
    """
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        H, W = img.shape
    
    h, w = H//2, W//2
    
    affin1 = np.array([[1, 0, -h], [0, 1, -w], [0, 0, 1]]) #まず画像中心と原点を一致させる
    affin2 = rotate_matrix1(theta)
    affin3 = np.array([[1, 0, h], [0, 1, w], [0, 0, 1]]) #最初にずらした分だけ逆方向に移動

    return np.matmul(affin3, np.matmul(affin2, affin1))



img = cv2.imread("imori.jpg")

out = affin_transform(img, rotate_matrix1(theta = -30.0))
cv2.imwrite("myanswer30img_1.jpg", out)
cv2.imshow("myanwer30img_1", out)
cv2.waitKey(0)
cv2.destroyWindow("out")


out = affin_transform(img, rotate_matrix2(img, theta = -30.0))
cv2.imwrite("myanswer30img_2.jpg", out)
cv2.imshow("myanwer30img_2", out)
cv2.waitKey(0)
cv2.destroyWindow("out_2")

cv2.destroyAllWindows()


# 注: 画像データはx座標が右方向, y座標が下方向に通っている
# 出力例を再現するには時計回りに30°(theta=-30)を入力する必要がある