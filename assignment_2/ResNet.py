# https://qiita.com/tchih11/items/377cbf9162e78a639958

## Assignment 9: Create VGG models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(0)  # For reproducibility

from typing import List, Union, Dict, Any

import torch
import torch.nn as nn


# 今回はvgg50のみ
# VGG.pyと同様ににconfigurationを読み込めるように拡張する予定
# cfgs: Dict[str, List[Union[str, int]]] = {
#     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "A_LRN": [64, "L", "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "C": [64, 64, "M", 128, 128, "M", 256, 256, "C", "M", 512, 512, "C", "M", 512, 512, "C", "M"],
#     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
#     "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
# }


import torch
import torch.nn as nn

class block(nn.Module): #モジュールのサブクラス
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        """
        Basicブロック(残差ブロック)を作成するクラス
        1. inputとoutputでchannel数、sizeが同じ
        2. outputのchannel数がinputの4倍
        3. outputのchannel数がinputの4倍、且つ、outputのsizeがinputの1/2
        の3つのパターンに対応している


        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : channel数調整用のconv層
            stride : 3×3conv層におけるstide数。sizeを半分にしたいときは2に設定
        """        
        super(block, self).__init__()

        # 1番目のconv層（1×1）
        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

        # 2番目のconv層（3×3）
        # パターン3の時はsizeを変更できるようにstrideは可変
        self.conv2 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels)

        # 3番目のconv層（1×1）
        # output channelはinput channelの4倍になる
        self.conv3 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(first_conv_out_channels*4)
        self.relu = nn.ReLU()

        # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
        self.identity_conv = identity_conv

    def forward(self, x):

        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 1×1の畳み込み
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 1×1の畳み込み
        x = self.bn3(x)

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()

        # conv1はアーキテクチャ通りにベタ打ち
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_xはサイズの変更は不要のため、strideは1
        self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

        # conv3_x以降はサイズの変更をする必要があるため、strideは2
        self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128, stride=2)
        self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256, stride=2)
        self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self,x):

        x = self.conv1(x)   # in:(3,224*224)、out:(64,112*112)
        x = self.bn1(x)     # in:(64,112*112)、out:(64,112*112)
        x = self.relu(x)    # in:(64,112*112)、out:(64,112*112)
        x = self.maxpool(x) # in:(64,112*112)、out:(64,56*56)

        x = self.conv2_x(x)  # in:(64,56*56)  、out:(256,56*56)
        x = self.conv3_x(x)  # in:(256,56*56) 、out:(512,28*28)
        x = self.conv4_x(x)  # in:(512,28*28) 、out:(1024,14*14)
        x = self.conv5_x(x)  # in:(1024,14*14)、out:(2048,7*7)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
        layers = []

        # 1つ目の残差ブロックではchannel調整、及びsize調整が発生する
        # identifyを足す前に1×1のconv層を追加し、サイズ調整が必要な場合はstrideを2に設定
        identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        # 2つ目以降のinput_channel数は1つ目のoutput_channelの4倍
        in_channels = first_conv_out_channels*4

        # channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)






if __name__=='__main__':

    
    x = torch.randn(32,3,244,244)  # input butch
    model = ResNet(block, 3) #今回は3クラス
    y = model(x)
    print(y.shape) #tensor(32,3)


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
    
    def show_datapoint(self, image, label): #画像データとラベルを受け取るように変更

        plt.imshow(image)
        plt.title("label: {}".format(label) )
        plt.show()

        return 0

transforms = Compose([
    Resize((128, 128)),
    ToTensor()
])

train_dataset = Cifar10Dataset(csv_file="cifar-10/train_3class_mini.csv", root_dir="cifar-10/images", transform=transforms)
test_dataset = Cifar10Dataset(csv_file="cifar-10/test_3class_mini.csv", root_dir="cifar-10/images", transform=transforms)
print("len(train_dataset):", len(train_dataset))
print("len(test_dataset):", len(test_dataset))





# Assignment 10: Apply to train data
import datetime  # <1>

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader):
    history_losses = np.zeros(n_epochs)
    history_accuracies = np.zeros(n_epochs)
    
    for i in range(n_epochs):  # <2>
        loss_total = 0.0
        count = 0
        for imgs, labels in train_loader:  # <3>
            
            
            outputs = model(imgs)  # <4>
            
            
            loss = loss_fn(outputs, labels)  # <5>

            optimizer.zero_grad()  # <6>
            
            loss.backward()  # <7>
            
            optimizer.step()  # <8>

            loss_total += loss.item()  # <9>
            count += 1
            print(count, "x32枚", loss_total)


        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            accuracy = correct / total



        print('{} Epoch {}, Training loss {}, testaccurasy {}'.format(
            datetime.datetime.now(), i+1,
            loss_total, accuracy))
        print()

        history_losses[i] = loss_total
        history_accuracies[i] = accuracy







    fig, axes = plt.subplots(1,2, figsize=(8,4), tight_layout=True, squeeze=False, facecolor = "white")
    fig.suptitle("ResNet 50")
    axes[0, 0].plot( history_losses, color='blue', marker='o', linestyle='dashed' )
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title("train losses")
    axes[0, 1].plot( history_accuracies, color='blue', marker='o', linestyle='dashed' )
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('loss')
    axes[0, 1].set_title("test accuracies")

    plt.savefig("out_imgs/ResNet_50.png")
    plt.show()





# train and validate
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32,shuffle=True) 
print("len(train_loader):",len(train_loader))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32,shuffle=True)
print("len(test_loader):",len(test_loader))


model = ResNet(block=block,num_classes=3) 
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
