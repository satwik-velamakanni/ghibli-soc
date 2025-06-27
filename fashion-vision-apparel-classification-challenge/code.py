import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Extract features and labels
X_train_np = train_df.iloc[:, 1:].values
y_train_np = train_df.iloc[:, 0].values

X_test_np  = test_df.iloc[:, :784].values
# Assuming test set also has labels in 785th column
y_test_np  = test_df.iloc[:, 784].values

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)

X_test_tensor  = torch.tensor(X_test_np, dtype=torch.float32)
print(X_train_tensor.shape)  # Should print: torch.Size([42000, 784])

# Create 4-fold splits for cross-validation
num_samples = X_train_tensor.shape[0]
fold_len = num_samples // 4
folds = []

for k in range(4):
    val_start = k * fold_len
    val_end = (k + 1) * fold_len if k < 3 else num_samples

    X_val_fold = X_train_tensor[val_start:val_end]
    y_val_fold = y_train_tensor[val_start:val_end]

    X_train_fold = torch.cat((X_train_tensor[:val_start], X_train_tensor[val_end:]), dim=0)
    y_train_fold = torch.cat((y_train_tensor[:val_start], y_train_tensor[val_end:]), dim=0)

    folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

# Setup training configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 500
output_dim = len(np.unique(y_train_np))
num_epochs = 100
train_batch = 100
test_batch = 64
learning_rate = 0.01

# Final DataLoader (no cross-val in this setup)
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data  = TensorDataset(X_test_tensor)

train_loader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=True)
test_loader  = DataLoader(dataset=test_data, batch_size=test_batch, shuffle=False)

# Peek at a batch
sample_batch = next(iter(test_loader))
sample_images = sample_batch[0]

# Define the neural network model
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = MLP(input_dim, hidden_dim, output_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_steps = len(train_loader)

for epoch in range(num_epochs):
    for step, (batch_imgs, batch_labels) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        preds = model(batch_imgs)
        loss = loss_fn(preds, batch_labels)

        # Backward and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item():.4f}")
