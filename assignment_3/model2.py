# reference: https://qiita.com/tchih11/items/377cbf9162e78a639958
# model2.py: ResNet18


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(0)  # For reproducibility

class block(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        """
        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : 残差接続時channel数調整用のconv層
            stride : 3×3conv層におけるstide数。出力sizeを半分にしたいときは2に設定。各レイヤの最初でしか用いない
        """        
        super(block, self).__init__()

        # conv層（stride指定時は出力サイズが半分に）
        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

        # conv層 (入力サイズと出力サイズが同じ)
        self.conv2 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels)
        
        # ReLU層
        self.relu = nn.ReLU()

        # 残差接続時、channel数の調整(各レイヤの最初)が必要な場合はconv層（1×1）を用意、不要な場合はNone
        self.identity_conv = identity_conv

    def forward(self, x):
        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 3×3の畳み込み
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # 3×3の畳み込みの繰り返し
        x = self.bn2(x)

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから残差を接続する
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()

        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_xは最初ブロックでサイズを変更しないため、strideは1
        self.conv2_x = self._make_layer(block, 2, res_block_in_channels=64, first_conv_out_channels=64, stride=1)
    

        # conv3_x以降は最初ブロックでサイズを変更する必要があるため、strideは2
        self.conv3_x = self._make_layer(block, 2, res_block_in_channels=64,  first_conv_out_channels=128, stride=2)
        self.conv4_x = self._make_layer(block, 2, res_block_in_channels=128,  first_conv_out_channels=256, stride=2)
        self.conv5_x = self._make_layer(block, 2, res_block_in_channels=256, first_conv_out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

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
        """
        このstrideは最初の残差ブロック(size調整ありのもの)でしか用いない
        """
        layers = []


        # 最初の残差ブロックでchannel調整、及びsize調整が発生する
        # 残差接続用に1×1のconv層を追加し、size調整が必要な場合(comv3以降)はstrideを2に設定
        identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        # 2つ目以降のinput_channel数は最初のブロックのoutput_channelに等しい
        in_channels = first_conv_out_channels
        # 以降channel調整、size調整は発生しないため、identity_convはNone、strideは1
        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)


if __name__=='__main__':

    
    x = torch.randn(5,3,224, 224)  # butchsize=5
    model = ResNet(block, 10)
    y = model(x)
    print(y)
    print(y.shape) #tensor(5,10)

    torch.save(model.state_dict(), "out_data/"+"unkomodel.pt")