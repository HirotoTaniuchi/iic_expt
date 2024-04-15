# Assignment 8: Dataloader with ToTensor and Normalize

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Import modules of transformations
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, ToTensor
from torchvision.transforms.functional import normalize

# Define Dataset class
class Cifar10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
            
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx): 
        """
        (変換済みのimage,label)のタプルを返すように変更する
        辞書を返す今までの仕様ではdataloaderの要求する引数に合わない
        """
        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        pilImg = Image.fromarray(image)
        transformed_img = self.transform(pilImg)
        #out = np.asarray(transformed_img) #ToTensorを使ったのでnp.array型に戻すとエラー
        label= self.landmarks_frame.iloc[idx, 1:]["label"]
        
        sample = (transformed_img, label)

        return sample
    
    # Utilities
    def show_datapoint(self, image, label): #画像データとラベルを受け取るように変更

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.savefig("out_imgs/out.png")
        plt.show()

        return 0




transforms = Compose([
    Resize((128, 128)),
    ToTensor()
])
#transformsにNormalizeを挟んでもバッチ正規化にはならない
#正則化は　(元のデータ – 平均) / (標準偏差)　で求まる  #各バッチの平均、標準偏差にそわせる必要がある
#transform ver.: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
#function ver. : https://pytorch.org/vision/main/generated/torchvision.transforms.functional.normalize.html


# Instanciate dataset class
dataset = Cifar10Dataset(csv_file="cifar-10/train.csv", root_dir="cifar-10/images", transform=transforms)


# Create dataloader
n_batch = 5
loader = DataLoader(dataset=dataset, batch_size=n_batch, shuffle = True)
print(loader)



# Get a mini-batch
for images, labels in loader:
    
    # Print num of images
    #print('images.shape = {}'.format(images.shape))
    #print("images.dtype = {}".format(images.dtype))
    
    # Calculate mean and std
    mean = images.mean(dim = (0,2,3)) #無くしたい次元をタプルで指定できる
    std = images.std(dim = (0,2,3))
    print("before_mean:", mean)
    print("before_std:", std)

    for i in range(n_batch):
        normalize(images[i], mean, std)


    print("after_mean:", images.mean(dim = (0,2,3)))
    print("after_std:", images.std(dim = (0,2,3)))

    break
