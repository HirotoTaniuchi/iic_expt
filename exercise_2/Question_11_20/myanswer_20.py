import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("imori_dark.jpg")
print(img.dtype)

fig = plt.figure()
# https://magazine.techacademy.jp/magazine/22285 #savefig()について
a = np.array(img)
plt.hist(a.flatten(), bins=np.arange(256 + 1))
#flatten()は一次元化, binsはヒストグラムの階級数
#https://qiita.com/zigenin/items/d93f2b5f28d4d227f349 #hist()について
plt.show()
fig.savefig("myanswer20img.jpg")
