import dataset #自作モジュール
from torch.utils.data import Dataset
import pandas as pd
import pathlib
import skimage.io
import PIL.Image
import matplotlib.pyplot as plt


import model #自作モジュール
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


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    history_losses = np.zeros(n_epochs)
    history_val = np.zeros(n_epochs)

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")
    model = model.to(device=device) #モデル内の重みもGPUに移す

    
    for i in range(n_epochs): 
        loss_total = 0.0
        count = 0


        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            

            outputs = model(imgs)

            loss = loss_fn(outputs, labels) 

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            loss_total += loss.item()
            count += 1

            if count%10==1:
                print("Epoch={}, batch={}(* batch_size)".format(i+1, count))


        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
            
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            accuracy = correct / total
        
        print("Epoch={}, train loss= {}, val accuracy={}".format(i+1, loss_total, accuracy))


        history_losses[i] = loss_total
        history_val[i] = accuracy
        np.save("out_data/ResNet_50_loss", history_losses)
        np.save("out_data/ResNet_50_val", history_val)





    fig, axes = plt.subplots(1,2, figsize=(8,4), tight_layout=True, squeeze=False, facecolor = "white")
    fig.suptitle("ResNet 50")
    axes[0, 0].plot(np.arange(1,n_epochs+1), history_losses, color='blue', marker='o', linestyle='dashed' )
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title("train loss")
    axes[0, 1].plot(np.arange(1,n_epochs+1), history_val, color='blue', marker='o', linestyle='dashed' )
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('accuracy')
    axes[0, 1].set_title("val accuracy")

    plt.savefig("out_data/ResNet_50.png")
    plt.show()


    torch.save(model.state_dict(), "out_data/"+"ResNet_50.pt")


if __name__=='__main__':

    transforms = Compose([Resize((224, 224)),ToTensor()]) #必須のtransform #正方形ではない画像データに対してはリサイズ必須
    dataset_train = dataset.KuzushiDataset(csv_file_path="data/multiple/train.txt", root_dir="data/img", transform = transforms)
    print("len(dataset_train):",len(dataset_train))
    # dataset_trainで用いたクラスとラベルの対応を継承する
    dataset_val = dataset.KuzushiDataset(csv_file_path="data/multiple/val.txt", root_dir="data/img", transform = transforms, class_to_label=dataset_train.class_to_label.copy())
    print("len(dataset_val):",len(dataset_val))

    # train and validate
    #batch大きすぎるとメモリ不足になる
    train_loader   = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=32,shuffle=True) 
    print("len(train_loader):",len(train_loader  ))
    val_loader   = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=32,shuffle=True) 
    print("len(val_loader):",len(val_loader  ))

    model   = model.ResNet(block=model.block, num_classes=10) 
    optimizer   = torch.optim.SGD(model  .parameters(), lr=1e-2)
    loss_fn   = nn.CrossEntropyLoss() 

    training_loop( 
        n_epochs = 100,
        optimizer = optimizer  ,
        model = model  ,
        loss_fn = loss_fn  ,
        train_loader = train_loader  ,
        val_loader = val_loader  
    )
