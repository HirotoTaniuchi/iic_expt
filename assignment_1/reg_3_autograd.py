# Regression using Tensor with Autograd
import torch 
torch.manual_seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

# Create random input
x = torch.randn((N,D_in))
t = torch.randn((N,D_out))

# Weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

# Train
for i in range(51):

    w1.requires_grad = True
    w2.requires_grad = True
    # Forward
    y = torch.matmul( torch.clamp( torch.matmul( x, w1), 0), w2)
    
    # Loss
    loss = torch.square(y - t).sum()
    if i%10 ==0:
        print(i, loss.item() )

    # Backward
    loss.backward()
     
    # Update weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Zero the gradients
        w1.grad.zero_()
        w2.grad.zero_()
