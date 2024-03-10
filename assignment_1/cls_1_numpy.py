import numpy as np
np.random.seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Input
x = np.random.randn(N, D_in)

# Target
labels = np.random.randint( 0, D_out - 1, N )  # class label
t = np.zeros((N,D_out))
t[range(N),labels] = 1  # target

# Weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
learning_rate = 1e-4

# Train
for i in range(51):
    # Forward
    y1 = np.matmul(x,w1)  
    y2 = np.maximum(y1, 0)  # ReLU
    y3 = np.matmul(y2,w2)  # (N,H) * (H,D_out) = (N,D_out)

    ## Softmax
    m = y3.max(axis=1,keepdims=True)
    y3_exp = np.exp(y3 - m)  # For avoiding inf
    y3_softmax = y3_exp / np.sum(y3_exp,axis=1,keepdims=True)

    # Cross entropy loss
    y = y3_softmax[range(N),labels]
    y[y==0.0] = 1e-100  # avoind inf
    loss = - np.sum(np.log(y))

    
    if i%10 == 0:
        print(i, loss)

    # Backward
    grad_y3 = y3_softmax.copy()
    grad_y3[range(N),labels] -= 1.0
    grad_w2 = y2.T.dot(grad_y3) 
    grad_y2 = grad_y3.dot(w2.T)
    grad_y1 = grad_y2.copy()
    grad_y1[y1 < 0] = 0
    grad_w1 = x.T.dot(grad_y1)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
