# Assignment 6: Dataset with image transformation

from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


# Define Dataset class
class Cifar10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
            
    def __len__(self):
        """
        __len__()はlen(x)で呼び出される。
        """
        return len(self.landmarks_frame)

    def __getitem__(self, idx): 
        """
        __getitem__はオブジェクトに角括弧でアクセスした時の挙動を定義する特殊メソッド
        ここで変換を加えることにする
        """
        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        pilImg = Image.fromarray(image)
        transformed_img = self.transform(pilImg)
        out = np.asarray(transformed_img)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        
        sample = {"image": out, "landmarks": landmarks}

        return sample
    
    # Utilities
    def show_datapoint(self, image, label): #画像データとラベルを受け取るように作り変える

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.savefig("out_imgs/out_6.png")
        plt.show()

        return 0
    
            


# Import modules of transformations
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomRotation

# Instanciate dataset class
transforms = Compose([
    Resize((128, 128)),
    RandomRotation((0, 90))
])
dataset = Cifar10Dataset(csv_file="cifar-10/train.csv", root_dir="cifar-10/images", transform=transforms)

# Get datapoint
rng = np.random.default_rng() #np.random.randintz()
randidx = rng.integers(0, dataset.__len__())
datapoint = dataset[randidx]

# Show
#print(datapoint["landmarks"]) #"landmarks"にはラベルとクラスとファイル名が全部入ってる
dataset.show_datapoint(image = datapoint["image"], label = datapoint["landmarks"]["label"])