# Assignment 5: Image transformation using torchvision

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
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        sample = {"image": image, "landmarks": landmarks}

        return sample
    
    # Utilities
    def show_datapoint(self, image, label): #画像データとラベルを受け取るように作り変える

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.savefig("out_imgs/out_5.png")
        plt.show()

        return 0
    



# Instanciate dataset class
dataset = Cifar10Dataset("cifar-10/train.csv", "cifar-10/images")

# Get datapoint
rng = np.random.default_rng() #np.random.randintz()
randidx = rng.integers(0, dataset.__len__())
datapoint = dataset[randidx]

# Show datapoint
#dataset.__show_datapoint__(randidx)




# Import modules of transformations
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip

# Convert to Image
# https://white-wheels.hatenadiary.org/entry/20100322/p1
pilImg = Image.fromarray(datapoint["image"])

# Transformation
# https://pytorch.org/vision/stable/transforms.html
transforms = Compose([
    Resize((128, 128)),
    RandomRotation((0, 90))
])
transformed_img = transforms(pilImg)

# Convert to array
out = np.asarray(transformed_img)

# Show
#print(datapoint["landmarks"]) #"landmarks"にはラベルとクラスとファイル名が全部入ってる
dataset.show_datapoint(image = out, label = datapoint["landmarks"]["label"])