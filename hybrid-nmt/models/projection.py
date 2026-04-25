"""
Projection Bridge: Maps IndicBERT output (768) to IndicTrans2 decoder input (1024)
"""

import torch
import torch.nn as nn


class ProjectionBridge(nn.Module):
    """
    Linear projection bridge with layer normalization and activation.
    
    Maps encoder output dimension (768) to decoder input dimension (1024).
    This is the ONLY component trained from scratch in Phase 1.
    
    Architecture: Linear(768 → 1024) → LayerNorm(1024) → GELU → Dropout(0.1)
    """
    
    def __init__(self, input_dim=768, output_dim=1024, dropout=0.1):
        """
        Args:
            input_dim (int): Input dimension (IndicBERT output)
            output_dim (int): Output dimension (IndicTrans2 decoder input)
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main projection components
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, encoder_hidden_states):
        """
        Project encoder output to decoder input dimension.
        
        Args:
            encoder_hidden_states (torch.FloatTensor): [batch_size, seq_len, 768]
        
        Returns:
            torch.FloatTensor: [batch_size, seq_len, 1024]
        """
        # Project: 768 → 1024
        projected = self.linear(encoder_hidden_states)
        
        # Apply layer norm, activation, and dropout
        projected = self.layer_norm(projected)
        projected = self.activation(projected)
        projected = self.dropout(projected)
        
        return projected
    
    def get_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
