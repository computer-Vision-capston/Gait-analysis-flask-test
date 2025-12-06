import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GaitLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, 
                 dropout=0.5, bidirectional=False):
        super(GaitLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class GaitLSTMTrainer:
    def __init__(self, data_dir='processed_data/preprocessed'):
        self.data_dir = Path(data_dir)
        
        X_train = np.load(self.data_dir / 'X_train.npy')
        X_val = np.load(self.data_dir / 'X_val.npy')
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_val = np.load(self.data_dir / 'y_val.npy')
        
        with open(self.data_dir / 'meta.json', 'r') as f:
            self.meta = json.load(f)
        
        # (samples, 400, 33, 4) -> (samples, 400, 132)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
        
        self.train_dataset = GaitDataset(X_train, y_train)
        self.val_dataset = GaitDataset(X_val, y_val)
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def build_model(self, hidden_size=64, num_layers=2, dropout=0.5,
                   bidirectional=False, learning_rate=0.001):
        input_size = 132
        
        self.model = GaitLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )
        
        return self.model
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training', leave=False)
        
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        return total_loss / len(dataloader), correct / total
    
    def train(self, batch_size=8, epochs=80, early_stopping_patience=20, save_every_n_epochs=5):
        if self.model is None:
            raise ValueError("build_model()을 먼저 실행하세요")
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), models_dir / 'best_model.pt')
            else:
                patience_counter += 1
            
            if epoch % save_every_n_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, models_dir / f'checkpoint_epoch_{epoch:02d}.pt')
            
            if patience_counter >= early_stopping_patience:
                print(f"Early Stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load(models_dir / 'best_model.pt'))
        
        train_loss, train_acc = self.validate(train_loader)
        val_loss, val_acc = self.validate(val_loader)
        print(f"\nFinal - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    def save_final_model(self, save_path='models/final_model.pt'):
        if self.model is None:
            return
        torch.save(self.model.state_dict(), save_path)


def main():
    trainer = GaitLSTMTrainer()
    
    trainer.build_model(
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        bidirectional=False,
        learning_rate=0.001
    )
    
    trainer.train(
        batch_size=8,
        epochs=80,
        early_stopping_patience=20,
        save_every_n_epochs=5
    )
    
    trainer.save_final_model()


if __name__ == "__main__":
    main()