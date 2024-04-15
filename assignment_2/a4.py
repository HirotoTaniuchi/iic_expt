# Assignment 4: Create dataset

from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
# from skimage.io import imread
import matplotlib.pyplot as plt

# Define Dataset class
class Cifar10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
            
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        #パスとファイル名を結合 #Pathはos.pathよりも便利
        #https://qiita.com/studio_haneya/items/11c9e825bd8068af7e87
        #img_name = self.root_dir + "/" + self.landmarks_frame.iloc[idx, 0]
        #.ilocはcolumnを整数(index)で指定できる
        #.locならcolumn名で指定

        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:] #そのインデックスのデータを全て取得(左端はインデックス自体なのでいらない)
        #landmarks = np.array([landmarks]) 
        #landmarks = landmarks.reshape(-1, 2) #referenceでreshapeする意味??
        sample = {"image": image, "landmarks": landmarks}

        return sample
    
    # Utilities
    def __show_datapoint__(self, idx):

        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        #subplotを使わない方法
        plt.imshow(image)
        plt.title("class: {}".format(self.landmarks_frame["class"][idx] ) )
        plt.savefig("out_imgs/out_4.png")
        plt.show()

        return 0


#アンダースコアの意味
#https://qiita.com/_Kohei_/items/069aa1e7b872f5ca96bf
#https://mako-note.com/ja/python-underscore/
        

# Instanciate dataset class
dataset = Cifar10Dataset("cifar-10/train.csv", "cifar-10/images")

# Get datapoint
rng = np.random.default_rng()
randidx = rng.integers(0, dataset.__len__())
item = dataset.__getitem__(randidx)
# print(item)

# Show datapoint
dataset.__show_datapoint__(randidx)