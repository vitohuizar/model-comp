from torch import nn


# ~~ Model ~~
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        input = 28*28
        hidden = 512
        output = 10

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    # Forward Pass
    def forward(self, x):
        x = self.flatten(x)
        out = self.layers(x)
        return out