# Regression using Numpy
import numpy as np
np.random.seed(0)  # For reproducibility

# N     : batch size
# D_in  : input dimension
# H     : hidden dimension
# D_out : output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

# Input and Target
x = np.random.randn(N, D_in)  # Input
t = np.random.randn(N, D_out)  # Target

# Weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Train
for i in range(51):
    # Forward
    y1 = np.matmul(x,w1)  
    y2 = np.maximum(y1, 0)  # ReLU
    y3 = np.matmul(y2,w2)

    # Square loss
    loss = np.square(y3 - t).sum()        
    if i%10 ==0:
        print(i, loss.item() )

    # Backward
    grad_y3 = 2.0 * (y3 - t)
    grad_w2 = y2.T.dot(grad_y3) 
    grad_y2 = grad_y3.dot(w2.T)
    grad_y1 = grad_y2.copy()
    grad_y1[y1 < 0] = 0
    grad_w1 = x.T.dot(grad_y1)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
