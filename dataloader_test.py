#%%
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

# set some params
dev = torch.device("cuda:0")
step_size = 0.01
loss_MBGD_20 = []
iter = 2
inputSize=1
outputSize=1

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, step_size).view(-1, 1).to(dev)
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = (-5 * X).to(dev)

class linmodel2(torch.nn.Module):
    def __init__(self, inputSize, outputSize, dev):
        super(linmodel2, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize,device=dev)
    def forward(self, x):
        out = self.linear(x)
        return out

# evaluating data points with Mean Square Error (MSE)
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Creating our dataset class
class Build_Data():
    # Constructor
    def __init__(self, dev, step_size):
        self.x = torch.arange(-5, 5, step_size).view(-1, 1).to(dev)
        self.y = (-5 * self.x + 0.4 * torch.randn(self.x.size(),device=dev)).to(dev)
        self.len = self.x.shape[0]
        # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        # Getting length of the data
    def __len__(self):
        return self.len

# Creating DataLoader object
dataset = Build_Data(dev, step_size)

# set up the data loader
train_loader_20 = DataLoader(dataset=dataset, batch_size=20, pin_memory=False)

# the model and optimizer
model = linmodel2(inputSize,outputSize,dev).to(dev)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, dampening=0.2, momentum=0.2)

# for each epoch
for i in range(iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_hat = model.forward(X)
    loss_MBGD_20.append(criterion(y_hat, Y).tolist())
    # then training on the data
    for x, y in train_loader_20:
        # making a prediction in forward pass
        y_hat = model.forward(x)
        # calculating the loss between original and predicted data points
        loss = criterion(y_hat, y)
        loss_MBGD_20.append(loss.tolist())
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()


plt.plot(loss_MBGD_20,label="Mini-Batch-20 Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/total loss')
plt.legend()
plt.show()


#%%
