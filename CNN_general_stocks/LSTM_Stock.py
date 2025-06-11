import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm

class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # output is a single price

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (batch, seq_len, hidden)
        out = out[:, -1, :]                # get last time step
        out = self.fc(out)
        return out.squeeze(-1)
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                val_loss = criterion(pred, y_batch)
                val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f}, Val Loss = {np.mean(val_losses):.4f}")
