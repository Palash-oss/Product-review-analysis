"""
Advanced Multimodal Sentiment Analysis Model
Architecture: Bi-Directional GRU with Attention Mechanism
Includes explainability features (attention weights for text highlighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionLayer(nn.Module):
    """Self-attention mechanism for capturing important words in text."""
    
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, gru_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_output: [batch_size, seq_len, hidden_dim]
        Returns:
            context_vector: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        # Calculate attention scores
        attention_scores = self.attention(gru_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # [batch_size, seq_len]
        
        # Apply attention weights
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            gru_output  # [batch_size, seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        return context_vector, attention_weights


class TextEncoder(nn.Module):
    """
    Bi-Directional GRU with Attention for text encoding.
    Provides attention weights for explainability.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
        # Project to shared latent space
        self.projection = nn.Linear(hidden_dim * 2, 256)
    
    def forward(self, text_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_input: [batch_size, seq_len] - tokenized text
        Returns:
            encoded: [batch_size, 256] - projected encoding
            attention_weights: [batch_size, seq_len] - for explainability
        """
        # Embedding
        embedded = self.embedding(text_input)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # GRU encoding
        gru_output, _ = self.gru(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Attention mechanism
        context_vector, attention_weights = self.attention(gru_output)
        
        # Project to shared space
        encoded = self.projection(context_vector)  # [batch_size, 256]
        encoded = F.relu(encoded)
        
        return encoded, attention_weights
    
    def load_pretrained_embeddings(self, embeddings: np.ndarray):
        """Load pretrained word embeddings (GloVe/FastText)."""
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))


class MultimodalSentimentModel(nn.Module):
    """
    Complete multimodal sentiment analysis model.
    Text-only mode with robust architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 3,  # Negative, Neutral, Positive
        dropout: float = 0.3
    ):
        super(MultimodalSentimentModel, self).__init__()
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(64, num_classes)
        
        # Regression head for continuous sentiment score
        self.regressor = nn.Linear(64, 1)
    
    def forward(
        self,
        text_input: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional attention weights for explainability.
        
        Args:
            text_input: [batch_size, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - logits: [batch_size, num_classes]
                - sentiment_score: [batch_size, 1] (continuous -1 to 1)
                - attention_weights: [batch_size, seq_len] (if return_attention=True)
        """
        # Encode text
        text_features, attention_weights = self.text_encoder(text_input)
        
        # Fusion
        fused_features = self.fusion(text_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Regression (continuous sentiment)
        sentiment_score = torch.tanh(self.regressor(fused_features))  # Scale to [-1, 1]
        
        output = {
            'logits': logits,
            'sentiment_score': sentiment_score,
            'probabilities': F.softmax(logits, dim=-1)
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def predict(
        self,
        text_input: torch.Tensor,
        return_explainability: bool = True
    ) -> Dict[str, any]:
        """
        High-level prediction method with explainability.
        
        Returns:
            Dictionary with predictions and explainability data
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(text_input, return_attention=return_explainability)
            
            # Get predicted class
            predicted_class = torch.argmax(output['logits'], dim=-1)
            confidence = torch.max(output['probabilities'], dim=-1)[0]
            
            result = {
                'predicted_class': predicted_class.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'sentiment_score': output['sentiment_score'].cpu().numpy(),
                'probabilities': output['probabilities'].cpu().numpy()
            }
            
            if return_explainability and 'attention_weights' in output:
                result['attention_weights'] = output['attention_weights'].cpu().numpy()
            
            return result


class SentimentTrainer:
    """Training utilities for the sentiment model."""
    
    def __init__(
        self,
        model: MultimodalSentimentModel,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Combined loss: classification + regression
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def train_step(
        self,
        text_input: torch.Tensor,
        labels: torch.Tensor,
        sentiment_scores: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        text_input = text_input.to(self.device)
        labels = labels.to(self.device)
        sentiment_scores = sentiment_scores.to(self.device)
        
        # Forward pass
        output = self.model(text_input)
        
        # Calculate losses
        cls_loss = self.classification_loss(output['logits'], labels)
        reg_loss = self.regression_loss(output['sentiment_score'], sentiment_scores.unsqueeze(1))
        
        # Combined loss
        total_loss = cls_loss + 0.5 * reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'classification_loss': cls_loss.item(),
            'regression_loss': reg_loss.item()
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
