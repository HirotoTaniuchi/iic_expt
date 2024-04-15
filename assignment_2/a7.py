# Assignment 7: Dataloader

from torch.utils.data import Dataset, DataLoader
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
        """
        (変換済みのimage,label)のタプルを返すように変更する
        辞書を返す今までの仕様ではdataloaderの要求する引数に合わない
        """
        img_name = Path(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        pilImg = Image.fromarray(image)
        transformed_img = self.transform(pilImg)
        out = np.asarray(transformed_img)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        
        sample = (out, landmarks["label"])

        return sample
    
    # Utilities
    def show_datapoint(self, image, label): #画像データとラベルを受け取るように変更

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.savefig("out_imgs/out_7.png")
        plt.show()

        return 0


# Import modules of transformations
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomRotation, ToTensor


# Instanciate dataset class
transforms = Compose([
    Resize((128, 128)),
    RandomRotation((0, 90))
    #ToTensor()
])
dataset = Cifar10Dataset(csv_file="cifar-10/train.csv", root_dir="cifar-10/images", transform=transforms)


# Create dataloader
n_batch = 5
loader = DataLoader(dataset=dataset, batch_size=n_batch, shuffle = True)


# Get a mini-batch
for images, labels in loader:
    
    # Print num of images
    print('images.shape = {}'.format(images.shape))
    print("images.dtype = {}".format(images.dtype))
    
    # Show images in batch
    fig, axes = plt.subplots(1,n_batch, figsize=(3*n_batch,3), tight_layout=True, squeeze=False)
    #matplotlibの出力では,permuteを利用して軸順序を CxHxW から HxWxC に変更する必要がある
    for i in range(n_batch):
        axes[0, i].imshow(images[i])
        #transformsにToTensorを使った場合は軸順序をplt用に修正してimages[i].permute(1, 2, 0)となる
        axes[0, i].set_title("label: " + str(labels[i].item())) #.item()でint型に, str()でstr型に
        #詰まり：print("a","b")と同じ感覚で何にでもカンマ使ってると、第2引数だと見なされてエラー出るから注意

    
    plt.savefig("out_imgs/out_7.png")
    plt.show()
    break




