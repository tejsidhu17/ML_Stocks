import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class StockCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_train_batch, y_train_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_train_batch)
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_train_batch.size(0)
            correct += (predicted == y_train_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_val_batch, y_val_batch in tqdm(val_loader, desc="Validation"):
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    outputs = model(X_val_batch)
                    loss = criterion(outputs, y_val_batch)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_val_batch.size(0)
                    val_correct += (predicted == y_val_batch).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

