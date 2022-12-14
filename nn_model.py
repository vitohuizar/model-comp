import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from cnn import ConvNeuralNet
from linear import NeuralNetwork


# Weights and Biases
import wandb
wandb.init(project="MNISTv2")

"""wandb.config = {
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 64
}"""


# Download training data from open datasets. 
train_data = datasets.MNIST(
    root="data",
    train=True, 
    download=True,
    transform=ToTensor()
    )


# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False, 
    download=True,
    transform=ToTensor()
    )


# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 100


# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Call Neural Network and load model state
model = ConvNeuralNet().to(device)
# model.load_state_dict(torch.load("models/model3.pth"))


# Optimizer & Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


# Training Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


        # Weights and Biases
        wandb.log({"loss": loss})
        wandb.watch(model)


# Test Loop
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Epochs
def training():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "models/model_cnn_dropout.pth")
    print("Done!")

if __name__ == "__main__":
    training()