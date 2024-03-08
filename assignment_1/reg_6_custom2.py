import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-8 # 1e-6だと大きすぎて発散してしまう

# Create random input
x = torch.randn((N,D_in))   # Input
t = torch.randn((N,D_out))  # Targets


def init_weights(m):
    "重みを標準正規分布で初期化する"
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)

# Model
model = nn.Sequential()
model.add_module("DxH", nn.Linear(D_in, H, bias = False))
# model.add_module("ReLU", nn.ReLU())
for i in range(98):
    model.add_module("HxH", nn.Linear(H, H, bias = False))
    # model.add_module("ReLU", nn.ReLU())
model.add_module("HxD", nn.Linear(H, D_out, bias = False))
# model.add_module("ReLU", nn.ReLU())
model.apply(init_weights)


# Loss function
loss_func = nn.MSELoss(reduction = "sum")

# Define optimizer
optimizer = optim.SGD( model.parameters(), lr=learning_rate)


print("100layers, lr=1e-8")
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







