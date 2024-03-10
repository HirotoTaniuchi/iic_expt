import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Input
x = torch.randn(N,D_in)

# Target
labels = torch.randint( 0, D_out-1, (N,) )  # class label
t = torch.zeros(N,D_out)
t[range(N),labels] = 1  # target

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
loss_func = nn.CrossEntropyLoss(reduction = "sum")

# Optimizer
learning_rate = 1e-4
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train
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