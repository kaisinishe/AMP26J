import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_model import OnsetCNN
from cnn_dataset import OnsetDataset
from data_loader import AMPDataLoader
from onset_detector import OnsetDetectorLFSF
import os

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Setup Data
    DATA_DIR = os.path.join("..", "data", "train", "train")
    loader = AMPDataLoader(DATA_DIR)
    detector = OnsetDetectorLFSF() # Shared config
    
    # Split tracks (80% train, 20% val)
    tracks = loader.track_ids
    train_num = int(0.8 * len(tracks))
    
    train_ds = OnsetDataset(tracks[:train_num], loader, detector)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = OnsetCNN().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    

    # 4. Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(windows).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()[0]}")
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # 5. Save the weights
    torch.save(model.state_dict(), "onset_cnn_v1.pth")
    print("Model saved to onset_cnn_v1.pth")

if __name__ == "__main__":
    train_model()