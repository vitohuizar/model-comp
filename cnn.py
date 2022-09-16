from torch import nn


class ConvNeuralNet(nn.Module):
    def __init__ (self):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Convolutional Layer
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            
            # Linear Layers
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x