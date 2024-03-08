# Regression using Tensor (without Autograd)
import torch
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

# Input and Target
x = torch.randn((N,D_in))   # Input
t = torch.randn((N,D_out))  # Target

# Weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

# Train
for i in range(51):
    # Forward
    y1 = torch.matmul(x, w1)
    y2 = torch.clamp(y1, 0)
    y3 = torch.matmul(y2, w2)
    
    # Loss
    loss = torch.square(y3 - t).sum()
    #print(i)
    if i%10 ==0:
        print(i, loss.item() )

    # Backward
    grad_y3 = 2.0 * (y3 - t)
    grad_w2 = torch.matmul(torch.transpose(y2, 0, 1), grad_y3)
    grad_y2 = torch.matmul(grad_y3, torch.transpose(w2, 0, 1))
    grad_y1 = torch.clone(grad_y2)
    grad_y1[y1 < 0] = 0
    grad_w1 = torch.matmul(torch.transpose(x, 0, 1), grad_y1)
    
    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

