import torch
import torch.nn as nn
import torchvision 
import numpy as np
import random
import matplotlib.pyplot as plt

def reprod_init():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
reprod_init()

mnist_train = torchvision.datasets.MNIST('./', download=True, train=True)
mnist_test = torchvision.datasets.MNIST('./', download=True, train=False)

x_train = mnist_train.train_data
y_train = mnist_train.train_labels
x_test = mnist_test.test_data
y_test = mnist_test.test_labels

x_train = x_train.float()
x_test = x_test.float()

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
