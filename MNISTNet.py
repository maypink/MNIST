import data 

class MNISTNet(nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        
        self.fc_block1=nn.Sequential(
        nn.Linear(784, n_hidden_neurons),
        nn.Sigmoid()
        )
        
        self.fc_block2=nn.Sequential(
        nn.Linear(n_hidden_neurons, 10)
        )

    def forward(self, x):
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        return x


mnist_net = MNISTNet(100)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)
