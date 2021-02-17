import numpy as np
import torch
import torch.nn as nn
from data import Data as d

def training(mnist_net):

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)

    test_loss_all = []
    test_accuracy_all = []
    batch_size = 300

    for epoch in range(3000):
        order = np.random.permutation(len(d.x_train))

        for i in range(0, len(d.x_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[i:i + batch_size]

            x_batch = d.x_train[batch_indexes]
            y_batch = d.y_train[batch_indexes]

            preds = mnist_net.forward(x_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = mnist_net.forward(d.x_test)
        test_loss_all.append(loss(test_preds, d.y_test))

        accuracy = (test_preds.argmax(dim=1) == d.y_test).float().mean()
        test_accuracy_all.append(accuracy)
        if epoch % 30 == 0 and epoch != 0:
            print('Average value of the loss function over the last 30 epochs is ', sum(test_loss_all[-30:]) / 30)
            print('Average value of the accuracy over the last 30 epochs is ', sum(test_accuracy_all[-30:]) / 30,
                  end='\n\n')

    return test_loss_all, test_accuracy_all

