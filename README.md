My PyTorch Learning Notes
These are my personal notes on what I’ve learned about using PyTorch to build and train neural networks. I’ve summarized the key concepts in simple terms to reflect my understanding.
PyTorch
I learned that PyTorch is a Python library for creating neural networks. It’s user-friendly and handles complex math operations efficiently, which makes it great for deep learning.
Tensors
Tensors are like arrays but optimized for calculations. I found they’re the core of PyTorch, used to store data like numbers or images.
Autograd
Autograd automatically calculates gradients, which are needed to optimize neural networks. I experimented with it and saw how it tracks operations on tensors:
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Got 4.0 as the gradient

Training Pipeline
I understood that training a neural network involves several steps:

Loading and preparing data.
Building a model.
Defining a loss function to measure errors.
Using an optimizer (like SGD or Adam) to update the model.
Training by feeding data and minimizing loss.
Testing the model’s performance.

NN Module
I used nn.Module to create neural networks by defining layers and how data flows through them. Here’s a simple model I tried:
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
    def forward(self, x):
        x = self.layer1(x)
        return x

Dataset
I learned to create a Dataset class to organize data. It defines how to load and preprocess data for PyTorch. Here’s an example I worked on:
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

DataLoader
I found that DataLoader helps load data in batches and can shuffle it. I tested it with my dataset:
from torch.utils.data import DataLoader

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

Training on GPU
I discovered that GPUs speed up training by handling many calculations at once. I learned to move models and data to the GPU like this:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = torch.tensor([1, 2, 3]).to(device)

Optimization of ANN
To improve my neural networks, I explored these techniques:
Dropouts
Dropouts prevent overfitting by randomly disabling neurons during training. I added it to a model:
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.dropout = nn.Dropout(0.2)  # Drops 20% of neurons
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        return x

Batch Regularization
I learned that batch normalization stabilizes and speeds up training by normalizing data in batches. I tried:
self.batch_norm = nn.BatchNorm1d(5)

Normalization
I found that normalizing data (e.g., scaling to 0–1) helps the model learn better. I used:
transform = transforms.Normalize(mean=0.5, std=0.5)

Hyperparameter Tuning
I learned that hyperparameters are settings like learning rate, batch size, number of epochs, or model architecture that affect how well the model performs. Tuning them means trying different values to find the best combination. I experimented with a grid search approach to test multiple learning rates and batch sizes, and I also tried different optimizers. Here’s the code I used to tune hyperparameters for a simple neural network (just added a part of the code which was actually responsible for hyperparameter tuning):
def objective(trial):

    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 80, 128, step=8)
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    input_dim = 784
    output_dim = 10

    model = MyNN(input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate)
    model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "SGD": 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to("cuda"), batch_labels.to("cuda")
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    total = correct = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to("cuda"), batch_labels.to("cuda")
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.shape[0]
            correct += (predicted == batch_labels).sum().item()
        accuracy = correct/total
    return accuracy

# use mlflow dashboard to visualise the best choice of parameters during hyperparameter tuning

What I Learned from Hyperparameter Tuning

I tested different learning rates (0.001, 0.01, 0.1) to see how fast the model learns without overshooting.
I tried batch sizes (16, 32, 64) to balance training speed and stability.
I compared optimizers (Adam and SGD) to understand their impact on convergence.
I ran the model for 10 epochs and tracked the loss to find the best combination.
This grid search helped me see which settings gave the lowest loss, but I realized it can be time-consuming. I could explore random search or libraries like Optuna for more efficient tuning in the future.

Transfer Learning
I learned that transfer learning uses pre-trained models (e.g., trained on ImageNet) and fine-tunes them for my task. It’s useful when I have limited data. I tested:
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

My Takeaways
I started by installing PyTorch (pip install torch torchvision) and experimenting with tensors. I built simple models with nn.Module, loaded data using Dataset and DataLoader, and trained on a GPU when available. Adding dropouts and batch normalization improved my models. Hyperparameter tuning was a big step for me—I learned how to systematically test different settings to optimize performance. Transfer learning was a cool way to use pre-trained models and save time. These notes capture my progress in learning PyTorch!
