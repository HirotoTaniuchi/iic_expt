
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# https://github.com/tosiyuki/vgg-food101
# https://note.com/toshi_456/n/n3a331a3ca767

## Assignment 9: Create VGG models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(0)  # For reproducibility

from typing import List, Union, Dict, Any

import torch
import torch.nn as nn


cfgs: Dict[str, List[Union[str, int]]] = { #unionは型ヒントで用いる
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "A_LRN": [64, "L", "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, "C", "M", 512, 512, "C", "M", 512, 512, "C", "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes=10, 
                 dropout: float=0.5, init_weights: bool=True):
        super(VGG, self).__init__()
        self.features = features
        self.avepool=nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential( #末尾の共通しているレイヤー
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avepool(x)
        x = torch.flatten(x, 1)#平坦化
        x = self.classifier(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]]):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            layers += [conv2d, nn.ReLU(True)]
        elif v == 'L':
            layers += [nn.LocalResponseNorm(5, k=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v

    return nn.Sequential(*layers) #リストをアンパック


# 乱数に適用してみる
if __name__=='__main__':
    x = torch.randn(32,3,244,244)  # n_butch=32の想定
    model = VGG(features = make_layers(cfgs["A"]), num_classes=3, dropout=0.5) #今回は3クラス
    y = model(x)
    #print(y) #tensor(32,3)





# make Dataset 

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Import modules of transformations
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.transforms.functional import normalize


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
        #out = np.asarray(transformed_img)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        
        sample = (transformed_img, landmarks["label"])

        return sample
    
    def show_datapoint(self, image, label):

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.show()

        return 0

transforms = Compose([
    Resize((128, 128)),
    ToTensor()
])

#学習時間の関係上,今回はクラス数を3つに削減したcsvファイルを用いることにする
#10クラス全てを用いると1epochの学習＆検証で45~60分くらいかかる
train_dataset = Cifar10Dataset(csv_file="cifar-10/train_3class.csv", root_dir="cifar-10/images", transform=transforms)
test_dataset = Cifar10Dataset(csv_file="cifar-10/test_3class.csv", root_dir="cifar-10/images", transform=transforms)
print("len(train_dataset):", len(train_dataset))
print("len(test_dataset):", len(test_dataset))








# Assignment 10: Apply to train data
import datetime 

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader):
    history_losses = np.zeros(n_epochs)
    history_accuracies = np.zeros(n_epochs)
    
    for i in range(n_epochs): 
        loss_total = 0.0
        count = 0
        for imgs, labels in train_loader:  
            
            
            outputs = model(imgs)  
            
            
            loss = loss_fn(outputs, labels)  

            optimizer.zero_grad() 
            
            loss.backward() 
            
            optimizer.step() 

            loss_total += loss.item() 
            count += 1
            print("epoch=", i," ", count, "x32枚", loss_total)


        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            accuracy = correct / total



        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), i+1,
            loss_total)) 
        print("and accuracy", accuracy)
        print()

        history_losses[i] = loss_total
        history_accuracies[i] = accuracy






    #show data
    fig, axes = plt.subplots(1,2, figsize=(8,4), tight_layout=True, squeeze=False, facecolor = "white")
    fig.suptitle("VGG A")
    axes[0, 0].plot( history_losses, color='blue', marker='o', linestyle='dashed' )
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title("train losses")
    axes[0, 1].plot( history_accuracies, color='blue', marker='o', linestyle='dashed' )
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('loss')
    axes[0, 1].set_title("test accuracies")

    plt.savefig("out_imgs/VGG_A.png")
    plt.show()







# train and validate
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32,shuffle=True)  
print("len(train_loader):",len(train_loader))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32,shuffle=True)  
print("len(train_loader):",len(test_loader))

model = VGG(features = make_layers(cfgs["A"]), num_classes=3, dropout=0.5) #今回は3クラス
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()  

training_loop(
    n_epochs = 10,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    test_loader = test_loader
)