"""
Transformer-based stress prediction model.

Adapted from translation transformer to encoder-only classification.
Reuses components: InputEmbeddings, PositionalEncoding, LayerNormalization,
FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection, EncoderBlock, Encoder.
"""

import torch
import torch.nn as nn
import math
import numpy as np


class InputProjection(nn.Module):
    """
    Project input features to d_model dimensions.
    Replaces embedding lookup used in translation.
    """
    def __init__(self, input_features, d_model):
        super().__init__()
        self.input_features = input_features
        self.d_model = d_model
        self.projection = nn.Linear(input_features, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, input_features)
        return self.projection(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    Adapted from translation transformer.
    """
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """Layer normalization with learnable parameters."""
    def __init__(self, features, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Feed-forward network with expansion and compression."""
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attention_scores = None  # For visualization
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        
        # Attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # Split into heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Compute attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """Residual connection with pre-normalization."""
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Single encoder block with self-attention and feed-forward."""
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    """Stack of encoder blocks."""
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class StressTransformerEncoder(nn.Module):
    """
    Encoder-only Transformer for stress classification.
    
    Adapted from translation transformer by:
    1. Replacing token embeddings with input projection layer
    2. Removing decoder entirely
    3. Adding classification head
    4. Using mean pooling for sequence aggregation
    """
    
    def __init__(self, 
                 input_features=65,
                 d_model=256,
                 N=4,
                 h=4,
                 d_ff=1024,
                 dropout=0.1,
                 seq_len=60,
                 pooling='mean'):
        """
        Args:
            input_features: Number of input features per timestep
            d_model: Embedding dimension
            N: Number of encoder layers
            h: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            seq_len: Sequence length
            pooling: Pooling strategy ('mean', 'max', 'cls')
        """
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.seq_len = seq_len
        self.pooling = pooling
        
        # Input projection (replaces embedding)
        self.input_projection = InputProjection(input_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        # Build encoder from blocks
        encoder_blocks = []
        for _ in range(N):
            self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, self_attention, feed_forward, dropout)
            encoder_blocks.append(encoder_block)
        
        self.encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_features)
            mask: Optional attention mask
        
        Returns:
            logits: Output logits (batch, 1)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder
        x = self.encoder(x, mask)  # (batch, seq_len, d_model)
        
        # Aggregate sequence
        if self.pooling == 'mean':
            x = x.mean(dim=1)  # (batch, d_model)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]  # (batch, d_model)
        elif self.pooling == 'cls':
            x = x[:, 0, :]  # Use first token (CLS token style)
        
        # Classify
        logits = self.classifier(x)  # (batch, 1)
        
        return logits
    
    def get_attention_weights(self):
        """Extract attention weights from all encoder blocks for visualization."""
        attention_weights = []
        for layer in self.encoder.layers:
            if hasattr(layer.self_attention_block, 'attention_scores'):
                attention_weights.append(layer.self_attention_block.attention_scores)
        return attention_weights


# Training class (similar to RNN trainer)
class StressTransformerTrainer:
    """
    Trainer for Transformer models.
    Same interface as RNN trainer for consistency.
    """
    
    def __init__(self, model, device='cpu', learning_rate=1e-4, pos_weight=1.0):
        """
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Learning rate (lower for transformers)
            pos_weight: Weight for positive class
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Optimizer (Adam with lower LR for transformers)
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
        
        for X_batch, y_batch in train_loader:
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
        
        return total_loss / len(train_loader)
    
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
                
                logits = self.model(X_batch)
                loss = self.criterion(logits.squeeze(), y_batch.float())
                
                total_loss += loss.item()
                
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
        """Train the model with early stopping."""
        best_val_auc = 0
        patience_counter = 0
        
        print(f"\nTraining {self.model.__class__.__name__}...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc, _, _ = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            self.scheduler.step(val_auc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val AUC = {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(self.best_model_state)
        
        print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_val_auc': best_val_auc
        }
    
    def predict(self, data_loader):
        """Make predictions."""
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
    # Test transformer architecture
    print("Testing Stress Transformer Encoder...")
    
    # Configuration
    batch_size = 32
    seq_len = 60
    input_features = 65
    
    # Create model
    model = StressTransformerEncoder(
        input_features=input_features,
        d_model=256,
        N=4,
        h=4,
        d_ff=1024,
        dropout=0.1,
        seq_len=seq_len,
        pooling='mean'
    )
    
    # Test forward pass
    X = torch.randn(batch_size, seq_len, input_features)
    output = model(X)
    
    print(f"\n✓ Model created successfully!")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test attention extraction
    attention_weights = model.get_attention_weights()
    print(f"  Attention layers: {len(attention_weights)}")
    if len(attention_weights) > 0 and attention_weights[0] is not None:
        print(f"  Attention shape: {attention_weights[0].shape}")
    
    print("\n✓ Transformer model tested successfully!")








