import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6
module_size = 100

# Create random input
x = torch.randn((N,D_in))   # Input
t = torch.randn((N,D_out))  # Targets


def init_weights(m):
    "重みを標準正規分布で初期化する"
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)

# Model
model = nn.Sequential()
#addmodluleの第一引数は名前。なお同じ名前にすると置き換えされてしまう
#バッチ正規化。入力の大きな値と小さな値の影響力を均す＝要素ごとに訓練データ集合を平均0かつ分散1にする。
model.add_module(f"norm {0}",nn.BatchNorm1d(D_in))
#線型変換
model.add_module(f"DxH  {0}", nn.Linear(D_in, H, bias = False))
#活性化関数ReLU。他の関数に比べ、微分の値が減衰せずに下層まで逆伝播してくれる
model.add_module(f"ReLU {0}", nn.ReLU())
for i in range(module_size-2):
    model.add_module(f"norm {i+1}",nn.BatchNorm1d(H))
    model.add_module(f"HxH  {i+1}", nn.Linear(H, H, bias = False))
    model.add_module(f"ReLU {i+1}", nn.ReLU())
model.add_module(f"HxD  {9}", nn.Linear(H, D_out, bias = False))
model.apply(init_weights)
#print(model)


# Loss function
loss_func = nn.MSELoss(reduction = "sum")

# Define optimizer
optimizer = optim.SGD( model.parameters(), lr=learning_rate)

print(f"{module_size}layers, learning_rate={learning_rate}")
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







