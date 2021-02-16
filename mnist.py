#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision 
import numpy as np
import random


# In[2]:


def reprod_init():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
reprod_init()


# In[3]:


mnist_train = torchvision.datasets.MNIST('./', download=True, train=True)
mnist_test = torchvision.datasets.MNIST('./', download=True, train=False)


# In[4]:


x_train = mnist_train.train_data
y_train = mnist_train.train_labels
x_test = mnist_test.test_data
y_test = mnist_test.test_labels


# In[5]:


print(x_train.dtype, y_train.dtype)


# In[6]:


x_train = x_train.float()
x_test = x_test.float()


# In[7]:


print(x_train.size())
print(y_train.size())
print(x_test.size())
print(y_test.size())


# In[8]:


x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape, x_test.shape)


# In[9]:


class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        #self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        #self.ac2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        #x = self.fc2(x)
        #x = self.ac2(x)
        x = self.fc3(x)
        return x


mnist_net = Net(100)


# In[10]:


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)


# In[ ]:


batch_size = 300
#order = np.random.permutation(len(x_train))

for epoch in range(3000):
    order = np.random.permutation(len(x_train))
    
    for i in range(0, len(x_train), batch_size):
        optimizer.zero_grad()
        
        batch_indexes = order[i:i+batch_size]
        
        x_batch = x_train[batch_indexes]
        y_batch = y_train[batch_indexes]
        
        preds = mnist_net.forward(x_batch) 
        
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        
        optimizer.step()

    test_preds = mnist_net.forward(x_test)
    
    #print(test_preds.argmax(dim=1))
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    print(accuracy)
        


# In[ ]:





# In[ ]:




