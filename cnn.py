from torch import nn


class ConvNeuralNet(nn.Module):
    def __init__ (self):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Convolutional Layer with dropout
            
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Linear Layers
            nn.Flatten(),
            nn.Linear(64*7*7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x