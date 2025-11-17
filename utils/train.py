import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 디바이스: {device}")


class GaitDataset(Dataset):
    """보행 데이터셋 클래스"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GaitLSTM(nn.Module):
    """보행 분류 LSTM 모델"""
    def __init__(self, input_size=88, hidden_size=64, num_layers=2, 
                 dropout=0.5, bidirectional=False):
        super(GaitLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense 레이어
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 hidden state 사용
        if self.bidirectional:
            # 양방향이면 forward와 backward 합치기
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        # Dense layers
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class GaitLSTMTrainer:
    def __init__(self, data_dir='processed_data/preprocessed'):
        """
        보행 분류 LSTM 모델 학습기 (PyTorch)
        
        Args:
            data_dir: 전처리된 데이터 폴더
        """
        self.data_dir = Path(data_dir)
        
        # 데이터 로드
        print("="*60)
        print("📂 데이터 로드 중...")
        print("="*60)
        
        X_train = np.load(self.data_dir / 'X_train.npy')
        X_val = np.load(self.data_dir / 'X_val.npy')
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_val = np.load(self.data_dir / 'y_val.npy')
        
        # 메타 정보 로드
        with open(self.data_dir / 'meta.json', 'r') as f:
            self.meta = json.load(f)
        
        print(f"✅ 학습 데이터: {X_train.shape}")
        print(f"✅ 검증 데이터: {X_val.shape}")
        print(f"   target_frames: {self.meta['target_frames']}")
        print(f"   num_keypoints: {self.meta['num_keypoints']}")
        
        # Shape 변환: (samples, 400, 33, 4) → (samples, 400, 132)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
        
        print(f"\n🔄 Shape 변환:")
        print(f"   학습: {X_train.shape} (samples, timesteps, features)")
        print(f"   검증: {X_val.shape}")
        
        # Dataset 생성
        self.train_dataset = GaitDataset(X_train, y_train)
        self.val_dataset = GaitDataset(X_val, y_val)
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def build_model(self, 
                   hidden_size=64,
                   num_layers=2,
                   dropout=0.5,
                   bidirectional=False,
                   learning_rate=0.001):
        """
        LSTM 모델 구축
        
        Args:
            hidden_size: LSTM hidden size
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율
            bidirectional: 양방향 LSTM 사용 여부
            learning_rate: 학습률
        """
        print("\n" + "="*60)
        print("🏗️  모델 구축 중...")
        print("="*60)
        
        input_size = 132  # 33 * 4
        
        self.model = GaitLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(device)
        
        # Optimizer & Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )
        
        print("\n📋 모델 구조:")
        print(self.model)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n📊 파라미터:")
        print(f"   전체: {total_params:,}")
        print(f"   학습 가능: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, dataloader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # tqdm 진행바 추가
        pbar = tqdm(dataloader, desc='Training', leave=False)
        
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            # Forward
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            # 진행바 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # tqdm 진행바 추가
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
                
                # 진행바 업데이트
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}'
                })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self,
             batch_size=8,
             epochs=80,
             early_stopping_patience=15,
             save_every_n_epochs=5):
        """
        모델 학습
        
        Args:
            batch_size: 배치 크기
            epochs: 최대 에포크 수
            early_stopping_patience: Early Stopping patience
            save_every_n_epochs: 체크포인트 저장 주기
        """
        if self.model is None:
            raise ValueError("모델을 먼저 build_model()로 생성하세요!")
        
        print("\n" + "="*60)
        print("🚀 학습 시작")
        print("="*60)
        
        # DataLoader 생성
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 모델 저장 폴더
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        print(f"\n⚙️  학습 설정:")
        print(f"   Batch size: {batch_size}")
        print(f"   Max epochs: {epochs}")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Checkpoint 저장 주기: {save_every_n_epochs} epochs")
        print(f"   모델 저장 위치: {models_dir}/")
        
        # Early stopping 변수
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 학습 루프
        for epoch in range(1, epochs + 1):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            
            # 현재 Learning rate 확인
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Learning rate 조정
            old_lr = current_lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 출력
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Learning rate 변경 알림
            if new_lr != old_lr:
                print(f"   📉 Learning Rate 감소: {old_lr:.6f} → {new_lr:.6f}")
            
            # 베스트 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), models_dir / 'best_model.pt')
                print(f"   ✅ 베스트 모델 저장 (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience: {patience_counter}/{early_stopping_patience}")
            
            # 주기적 체크포인트 저장
            if epoch % save_every_n_epochs == 0:
                checkpoint_path = models_dir / f'checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"   💾 체크포인트 저장: {checkpoint_path.name}")
            
            print()  # 빈 줄
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early Stopping! (Patience: {early_stopping_patience})")
                print(f"   Best Val Loss: {best_val_loss:.4f}")
                break
        
        print("\n" + "="*60)
        print("✅ 학습 완료!")
        print("="*60)
        
        # 베스트 모델 로드
        self.model.load_state_dict(torch.load(models_dir / 'best_model.pt'))
        
        # 최종 성능
        train_loss, train_acc = self.validate(train_loader)
        val_loss, val_acc = self.validate(val_loader)
        
        print(f"\n📊 최종 성능 (베스트 모델):")
        print(f"   Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    def plot_history(self, save_path='models/training_history.png'):
        """학습 과정 시각화"""
        if not self.history['train_loss']:
            print("⚠️  학습 기록이 없습니다!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n💾 학습 그래프 저장: {save_path}")
        plt.close()
    
    def save_final_model(self, save_path='models/final_model.pt'):
        """최종 모델 저장"""
        if self.model is None:
            print("⚠️  저장할 모델이 없습니다!")
            return
        
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 최종 모델 저장: {save_path}")


def main():
    """메인 실행"""
    # 학습기 생성
    trainer = GaitLSTMTrainer()
    
    # 모델 구축
    trainer.build_model(
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        bidirectional=False,
        learning_rate=0.001
    )
    
    # 학습
    trainer.train(
        batch_size=8,
        epochs=80,
        early_stopping_patience=15,
        save_every_n_epochs=5
    )
    
    # 결과 저장
    trainer.plot_history()
    trainer.save_final_model()
    
    print("\n" + "="*60)
    print("🎉 모든 작업 완료!")
    print("="*60)


if __name__ == "__main__":
    main()