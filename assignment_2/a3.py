# Assignment 3: Randomly show train images

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

# Randomly show images
fig, axes = plt.subplots(1,5, figsize=(14,3), tight_layout=True, squeeze=False)
#https://note.nkmk.me/python-numpy-random/#generator117
rng = np.random.default_rng()
randindexes = rng.integers(0, ntrain, size=5)

for i in range(5): #range(ntrain):
    # Load image
    fn = df.filename[randindexes[i]]
    path = "cifar-10/images/{}".format(fn)
    axes[0, i].imshow(imread(path))
    #print(df["class"][i])
    #print(df["label"][i])
    axes[0, i].set_title(fn + ", " + df["class"][randindexes[i]] + ", " + str(df["label"][randindexes[i]]))


plt.savefig("out_imgs/out_3.png")
plt.show()
