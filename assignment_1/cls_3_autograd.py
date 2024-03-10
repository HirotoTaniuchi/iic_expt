import torch
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

# Weights
w1 = torch.randn(D_in, H, requires_grad = True)  # Turn on autograd 
w2 = torch.randn(H, D_out, requires_grad = True)  # Turn on autograd
learning_rate = 1e-4

#
loss_func = torch.nn.CrossEntropyLoss(reduction = "sum")

for i in range(51):
    # Forward
    y = x.mm(w1).clamp(min=0).mm(w2)


    # Loss
    loss = loss_func(y, t)
    #nn.CrossEntropyLossにはsoftmax関数が内蔵されている
    #https://qiita.com/ground0state/items/8933f9ef54d6cd005a69
    
    if i%10 == 0:
        print(i, loss.item())

    # Backward
    loss.backward()

    # Update weight
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Zero the gradients
        w1.grad.zero_()
        w2.grad.zero_()