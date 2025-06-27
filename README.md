# My PyTorch Learning Notes

These are my notes on using PyTorch to build and train neural networks, summarized in simple terms.

## PyTorch
PyTorch is a Python library for neural networks. It’s easy to use and great for deep learning.

## Tensors
Tensors are arrays optimized for math, used to store data like images.

```python
import torch
x = torch.tensor([1, 2, 3])  # 1D tensor
```

## Autograd
Autograd computes gradients automatically for training. I tested it:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Outputs 4.0
```

## Training Pipeline
Training involves:
1. Preparing data.
2. Building a model.
3. Defining loss and optimizer.
4. Training and testing.

## NN Module
I used `nn.Module` to build networks by defining layers:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
    def forward(self, x):
        x = self.layer1(x)
        return x
```

## Dataset
I created a `Dataset` to organize data:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```

## DataLoader
`DataLoader` loads data in batches and shuffles it:

```python
from torch.utils.data import DataLoader

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

## Training on GPU
GPUs speed up training. I moved models/data to GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = torch.tensor([1, 2, 3]).to(device)
```

## Optimization of ANN
I used these to improve my networks:

### Dropouts
Dropouts prevent overfitting by disabling neurons:

```python
self.dropout = nn.Dropout(0.2)  # Drops 20% neurons
```

### Batch Regularization
Batch normalization stabilizes training:

```python
self.batch_norm = nn.BatchNorm1d(5)
```

### Normalization
Normalizing data (0–1 range) helps learning:

```python
transform = transforms.Normalize(mean=0.5, std=0.5)
```

## Hyperparameter Tuning
I used Optuna for hyperparameter tuning to optimize parameters like learning rate, batch size, and number of layers. Here’s the code I worked with:

```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow

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
        model.train()
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
        accuracy = correct / total
        
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}, Best accuracy: {study.best_value}")
```

I learned that Optuna efficiently tests combinations of hyperparameters, improving model accuracy compared to manual tuning.

## Transfer Learning
Transfer learning uses pre-trained models and fine-tunes them. I tried:

```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
```

## My Takeaways
I installed PyTorch (`pip install torch torchvision`) and worked with tensors, models, and data loading. I improved models with dropouts, normalization, and hyperparameter tuning using Optuna. Transfer learning saved time. These notes reflect my PyTorch journey!
