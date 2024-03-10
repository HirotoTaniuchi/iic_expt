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
w1 = torch.randn(D_in, H)  # Do not use autograd
w2 = torch.randn(H, D_out)  # Do not use autograd
learning_rate = 1e-6

# Train
for i in range(201):
    y1 = torch.matmul(x, w1)
    y2 = torch.clamp(y1, 0)
    y3 = torch.matmul(y2, w2)

    ## Softmax
    #m = torch.softmax(y3, 1)
    
    # Loss
    #loss_func = torch.nn.CrossEntropyLoss(reduction = "sum")
    #loss = loss_func(m, t)



    ## Softmax
    m = y3.max(axis=1,keepdims=True)[0]

    y3_exp = torch.exp(y3 - m)  # For avoiding inf

    y3_softmax = y3_exp / torch.sum(y3_exp,axis=1,keepdims=True)

    # Cross entropy loss
    y = y3_softmax[range(N),labels]
    y[y==0.0] = 1e-20  # avoind inf
    
    loss = - torch.sum(torch.log(y))


    
    #print(i)
    if i%10 ==0:
        print(i, "loss",loss)

    # Backward
    grad_y3 = torch.clone(y3)
    grad_y3[range(N),labels] -= 1.0
    grad_w2 = torch.matmul(torch.transpose(y2, 0, 1), grad_y3)
    grad_y2 = torch.matmul(grad_y3, torch.transpose(w2, 0, 1))
    grad_y1 = torch.clone(grad_y2)
    grad_y1[y1 < 0] = 0
    grad_w1 = torch.matmul(torch.transpose(x, 0, 1), grad_y1)
    
    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
