import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6 # 1e-6だと大きすぎて発散してしま
layer_size = 10

# Create random input
x = torch.randn((N,D_in))   # Input
t = torch.randn((N,D_out))  # Targets


def init_weights(m):
    "重みを標準正規分布で初期化する"
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)



#nn.Moduleを継承してサブクラスを定義する。その際nn.ModuleListも使う
#クラスの継承,init()についてhttps://tech.hipro-job.jp/column/771#:~:text=コンストラクタ
class Model(nn.Module):
    def __init__(self, n_layers:int = 10):
        #まずスーパークラスの初期化 https://qiita.com/Anaakikutsushit/items/c67d4edb0c01a97f3130
        super().__init__()
        self.firstlayers = nn.ModuleList( #アンパックの必要なし
            [nn.BatchNorm1d(D_in), #バッチ正規化（白色かの代わり）使うと収束が早くなる
            nn.Linear(D_in, H, bias = False),#bias無くすと収束が早くなる
            nn.ReLU()])
        self.middlelayers = nn.ModuleList(
            [nn.BatchNorm1d(H),
            nn.Linear(H, H, bias = False),
            nn.ReLU()] * (n_layers - 2))
        self.lastlayer = nn.Linear(H, D_out, bias = False)

    def forward(self, x):
        for layer in self.firstlayers:#モジュールリストはリスト形式で渡されるのでfor文
            x = layer(x)
        for layer in self.middlelayers:
            x = layer(x)
        x = self.lastlayer(x)
        return x



model = Model(layer_size) #サブクラスを用いてインスタンスを宣言
#これ以降は今までと同様に動作する
model.apply(init_weights) #重みの初期化


# Loss function
loss_func = nn.MSELoss(reduction = "sum")

# Define optimizer
optimizer = optim.SGD( model.parameters(), lr=learning_rate)



print(f"{layer_size}layers, learning_rate={learning_rate}")
for i in range(51):
    # Forward
    y = model(x)
    
    # Loss
    loss = loss_func(y, t)
    if i%10 ==0:
        print(i, loss.item() )

    # Backward
    loss.backward()# Backward
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()







