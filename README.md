# My PyTorch Learning Notes

These are my notes on using PyTorch to build and train neural networks, summarized in simple terms.

## PyTorch
PyTorch is a Python library for neural networks. It’s easy to use and great for deep learning.

## Tensors
Tensors are arrays optimized for math. They’re the core of PyTorch for storing data like images.

```python
import torch
x = torch.tensor([1, 2, 3])  # 1D tensor
```

## Autograd
Autograd computes gradients automatically for training. I tested it like this:

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
I created a `Dataset` to organize data for PyTorch:

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
I tuned hyperparameters like learning rate and batch size using grid search to optimize performance:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

learning_rates = [0.001, 0.01]
batch_sizes = [16, 32]
optimizers = [optim.Adam, optim.SGD]
epochs = 5

dataset = MyDataset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_loss = float('inf')
best_params = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for opt_class in optimizers:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model = MyModel().to(device)
            criterion = nn.MSELoss()
            optimizer = opt_class(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                total_loss = 0
                for data in dataloader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {'lr': lr, 'batch_size': batch_size, 'optimizer': opt_class.__name__}

print(f"Best parameters: {best_params}, Best loss: {best_loss}")
```

I learned that tuning learning rates, batch sizes, and optimizers improves model performance, but grid search takes time.

## Transfer Learning
Transfer learning uses pre-trained models and fine-tunes them. I tried:

```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
```

## My Takeaways
I installed PyTorch (`pip install torch torchvision`) and worked with tensors, models, and data loading. I improved models with dropouts, normalization, and hyperparameter tuning. Transfer learning saved time. These notes reflect my PyTorch journey!
