import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import pickle
import os

# ======================================================
# Paths (dev_v5)
# ======================================================
DATASET_DIR = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/datasets"
MODEL_SAVE_PATH = "C:/PERSONAL_DATA/Startups/Stocks/Jim_Simons_Trading_Strategy/AI_ML/ML/dev_v5/models/lstm_model.pth"
FEATURE_COLS_PICKLE = os.path.join(DATASET_DIR, "feature_cols.pkl")

# ======================================================
# 1️⃣ Load Data
# ======================================================
def load_dataset(dataset_dir):
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))

    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))

    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    with open(FEATURE_COLS_PICKLE, "rb") as f:
        feature_cols = pickle.load(f)

    print(f"Dataset shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# ======================================================
# 2️⃣ Prepare DataLoader
# ======================================================
def create_dataloader(X, y, batch_size=128, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ======================================================
# 3️⃣ LSTM Model
# ======================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # regression for future_return

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

# ======================================================
# 4️⃣ Training Loop
# ======================================================
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return model

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_dataset(DATASET_DIR)

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val, shuffle=False)

    input_size = X_train.shape[2]
    model = LSTMModel(input_size)

    print("Training model...")
    model = train_model(model, train_loader, val_loader, epochs=5, lr=0.001)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}")
