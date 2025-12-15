"""
RNN-based models (LSTM/GRU) for stress prediction.

Implements bidirectional LSTM and GRU models with classification heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressLSTM(nn.Module):
    """
    Bidirectional LSTM for stress prediction.
    
    Architecture:
    - Bidirectional LSTM layers
    - Dropout for regularization
    - Fully connected classification head
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(StressLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        # *2 because bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            logits: Output logits (batch_size, 1)
        """
        # LSTM forward pass
        # output: (batch, seq_len, hidden_size * 2)
        # h_n: (num_layers * 2, batch, hidden_size)
        # c_n: (num_layers * 2, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        # output[:, -1, :] gives (batch, hidden_size * 2)
        last_output = output[:, -1, :]
        
        # Classification
        logits = self.fc(last_output)
        
        return logits


class StressGRU(nn.Module):
    """
    Bidirectional GRU for stress prediction.
    
    Architecture:
    - Bidirectional GRU layers
    - Dropout for regularization
    - Fully connected classification head
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden layer size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(StressGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        # *2 because bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            logits: Output logits (batch_size, 1)
        """
        # GRU forward pass
        # output: (batch, seq_len, hidden_size * 2)
        # h_n: (num_layers * 2, batch, hidden_size)
        output, h_n = self.gru(x)
        
        # Use last timestep output
        last_output = output[:, -1, :]
        
        # Classification
        logits = self.fc(last_output)
        
        return logits


class StressRNNTrainer:
    """
    Trainer for RNN models.
    
    Handles training loop, validation, and early stopping.
    """
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, pos_weight=1.0):
        """
        Args:
            model: PyTorch model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            pos_weight: Weight for positive class in loss function
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function with class weighting
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits.squeeze(), y_batch.float())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                logits = self.model(X_batch)
                loss = self.criterion(logits.squeeze(), y_batch.float())
                
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.sigmoid(logits.squeeze())
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0
        
        return avg_loss, auc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=10):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        best_val_auc = 0
        patience_counter = 0
        
        print(f"\nTraining {self.model.__class__.__name__}...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc, _, _ = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val AUC = {val_auc:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_val_auc': best_val_auc
        }
    
    def predict(self, data_loader):
        """Make predictions on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits.squeeze())
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        return np.array(all_preds), np.array(all_labels)


if __name__ == '__main__':
    # Test model architectures
    print("Testing RNN models...")
    
    # Create dummy data
    batch_size = 32
    seq_len = 60  # 60 seconds at 1Hz
    input_size = 65  # Number of features
    
    X = torch.randn(batch_size, seq_len, input_size)
    
    # Test LSTM
    print("\nTesting LSTM...")
    lstm = StressLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    output = lstm(X)
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # Test GRU
    print("\nTesting GRU...")
    gru = StressGRU(input_size=input_size, hidden_size=128, num_layers=2)
    output = gru(X)
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gru.parameters()):,}")
    
    print("\n✓ All RNN models tested successfully!")








