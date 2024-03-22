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
layer_size = 10

# Create random input
x = torch.randn((N,D_in))   # Input
t = torch.randn((N,D_out))  # Targets


def init_weights(m):
    "重みを標準正規分布で初期化する＝誤差が大味になる"
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)



#リストを作成→アンパックの流れでスッキリさせる
first_layer = [nn.BatchNorm1d(D_in), #バッチ正規化（白色かの代わり）使うと収束が早くなる
               nn.Linear(D_in, H, bias = False),#bias無くすと収束が早くなる
               nn.ReLU()]
layers = [nn.BatchNorm1d(H),
          nn.Linear(H, H, bias = False),
          nn.ReLU()] * (layer_size - 2)
#print(layers)

model = nn.Sequential(
    *first_layer,#Sequentialにはリストをアンパックして中身だけ渡す
    *layers,
    nn.Linear(H, D_out, bias = False)
)
#print(model)
model.apply(init_weights)






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







