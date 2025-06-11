import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class MLPStockPredictor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(MLPStockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        return self.fc5(x)
    
def ordinal_encode(y, num_classes=5):
        return torch.stack([y > i for i in range(num_classes - 1)], dim=1).float()
    
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=50):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # correct_train = 0
        # total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            targets = ordinal_encode(y_batch, 5).to(device)
            # print("Logits shape:", logits.shape)  # Should be (batch_size, 4)
            # print("Targets shape:", targets.shape)  # Should be (batch_size, 4)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)

            # with torch.no_grad():
            #     preds = ordinal_encode(torch.sigmoid(logits))
            #     correct_train += (preds == y_batch).sum().item()
            #     total_train += len(y_batch)

        train_loss /= len(train_loader.dataset)
        # train_acc = correct_train / total_train

        model.eval()
        val_loss = 0
        # correct_val = 0
        # total_val = 0
        # val_preds = []
        # val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                targets = ordinal_encode(y_batch, 5).to(device)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(y_batch)

                # preds = ordinal_encode(torch.sigmoid(logits))
                # correct_val += (preds == y_batch).sum().item()
                # val_preds.extend(preds.cpu().numpy())
                # val_targets.extend(y_batch.cpu().numpy())
                # total_val += len(y_batch)
        val_loss /= len(val_loader.dataset)
        # val_acc = correct_val/total_val

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        # print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.6f}, Val Accuracy: {val_acc:.6f}")

    
