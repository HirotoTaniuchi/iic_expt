import dataset #自作モジュール
from torch.utils.data import Dataset
import pandas as pd
import pathlib
import skimage.io
import PIL.Image
import matplotlib.pyplot as plt


import model2 #自作モジュール
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(0)  # For reproducibility

#import torch
import datetime
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
#import matplotlib.pyplot as plt


def test(model, train_loader, test_loader):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Testing on device {device}.")

    for name, loader in [("train", train_loader), ("test", test_loader)]:

        correct = 0
        total = 0


        with torch.no_grad():
            for imgs, labels in loader:


                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                model = model.to(device=device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.5f}".format(name , correct / total))


if __name__ == "__main__":
    
    # 保存されているのはモジュールの重みとバイアスだけ
    # 重みを本番環境にデプロイするにはすぐに使えるモデルのクラスを読み込む必要がある
    transforms = Compose([Resize((224, 224)),ToTensor()])
    dataset_train = dataset.KuzushiDataset(csv_file_path="data/multiple/train.txt", root_dir="data/img", transform = transforms)
    print(dataset_train.class_to_label)
    print("len(dataset_train):", len(dataset_train))
    # dataset_trainで用いたクラスとラベルの対応を継承する
    dataset_test = dataset.KuzushiDataset(csv_file_path="data/multiple/test.txt", root_dir="data/img", transform = transforms, class_to_label=dataset_train.class_to_label.copy())
    print("len(dataset_test):", len(dataset_test))


    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=32,shuffle=True) 
    print("len(train_loader_1):",len(train_loader))
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=32,shuffle=True) 
    print("len(test_loader_1):",len(test_loader))

    loaded_model = model2.ResNet(block=model2.block, num_classes=10) 
    loaded_model.load_state_dict(torch.load("out_data/"+"ResNet_18.pt"))


    test(loaded_model, train_loader, test_loader)

    # ResNet18
    # Accuracy train: 0.90919
    # Accuracy test: 0.86590


    

