# Assignment 2: Show train images

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
    
# Load class info
class_path = "cifar-10/class.csv"
df = pd.read_csv(class_path)
labels = df["label"]
cls_names = df["class"]


# Load train.csv
train_path = "cifar-10/train.csv"
df = pd.read_csv(train_path)

# Print how many datapoints
ntrain = df.shape[0]
print('ntrain = {}'.format(ntrain))
#print(df.head())


# # Show images
fig, axes = plt.subplots(1,5, figsize=(14,3), tight_layout=True, squeeze=False)
for i in range(5): #range(ntrain):
    # Load image
    fn = df.filename[i]
    path = "cifar-10/images/{}".format(fn)
    axes[0, i].imshow(imread(path))
    #print(df["class"][i])
    #print(df["label"][i])
    axes[0, i].set_title(fn + ", " + df["class"][i] + ", " + str(df["label"][i]))


plt.savefig("out_imgs/out_2.png")
plt.show()


# # Show forの中でいちいちplt.show()してもok
# plt.imshow(img)
# plt.title(f'{fn}, {df.iloc[i]["class"]}, {df.iloc[i]["label"]}')
# plt.show()