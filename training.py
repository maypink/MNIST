import numpy as np
import torch
import torch.nn as nn
from data import data_prep

def training(mnist_net):

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)

    test_loss_all = []
    test_accuracy_all = []
    batch_size = 300

    x_train, y_train, x_test, y_test = data_prep()

    for epoch in range(3000):
        order = np.random.permutation(len(x_train))

        for i in range(0, len(x_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[i:i + batch_size]

            x_batch = x_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            preds = mnist_net.forward(x_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = mnist_net.forward(x_test)
        with torch.no_grad():
            test_loss_all.append(loss(test_preds, y_test))

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
        test_accuracy_all.append(accuracy)
        if epoch % 30 == 0 and epoch != 0:
            print('Average value of the loss function over the last 30 epochs is ', sum(test_loss_all[-30:]) / 30)
            print('Average value of the accuracy over the last 30 epochs is ', sum(test_accuracy_all[-30:]) / 30,
                  end='\n\n')

    return test_loss_all, test_accuracy_all

