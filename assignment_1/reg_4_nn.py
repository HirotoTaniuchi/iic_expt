# Regression using Torch and PyTorch modules
import torch
import torch.nn as nn
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

# Create random input
x = torch.randn((N,D_in))  # input
t = torch.randn((N,D_out))  # targets

#今までのコードは初期重みを乱数で初期重みを設定していた. 
#損失の遷移が同じ挙動するよう機能を追加
def init_weights(m):
    "重みを標準正規分布で初期化する"
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)

# Model
model = nn.Sequential(
    nn.Linear(D_in, H, bias = False),
    nn.ReLU(),
    nn.Linear(H, D_out, bias = False)
)
model.apply(init_weights)

# Loss function
loss_func = nn.MSELoss(reduction = "sum")

# Train
for i in range(51):
    # Forward
    y = model(x)
    
    # Loss
    loss = loss_func(y, t)
    if i%10 ==0:
        print(i, loss.item() )

    # Backward
    loss.backward()    # Backward
    
    # Update weights
    with torch.no_grad():
        for param in [model[0].weight, model[2].weight]:
            param -= learning_rate * param.grad # Zero the gradients of parameters
            param.grad.zero_()

