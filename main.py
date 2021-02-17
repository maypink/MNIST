from training import training
from data import reprod_init
from vizualization import vizualization
from MNISTNet import MNISTNet

reprod_init()

mnist_net = MNISTNet(100)

test_loss, test_accuracy = training(mnist_net)

vizualization(test_loss, test_accuracy)

