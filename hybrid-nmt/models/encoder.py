"""
IndicBERT v2 Encoder wrapper for the hybrid model
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class IndicBERTEncoder(nn.Module):
    """
    IndicBERT v2 encoder for the hybrid NMT model.
    
    Takes English text and produces contextual representations.
    Output dimension: 768
    """
    
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM"):
        """
        Args:
            model_name (str): HuggingFace model identifier
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.output_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the encoder.
        
        Args:
            input_ids (torch.LongTensor): [batch_size, seq_len]
            attention_mask (torch.LongTensor): [batch_size, seq_len]
        
        Returns:
            torch.FloatTensor: [batch_size, seq_len, 768]
                Last hidden state from the encoder
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Return last hidden state
        return outputs.last_hidden_state
    
    def freeze(self):
        """Freeze all encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, n=4):
        """
        Unfreeze the top N transformer layers.
        
        Args:
            n (int): Number of top layers to unfreeze
        """
        # Freeze everything first
        self.freeze()
        
        # Unfreeze top N layers
        # IndicBERT uses transformers.BertModel with encoder.layer[i] structure
        num_layers = len(self.model.encoder.layer)
        
        for i in range(num_layers - n, num_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze layer norm and embeddings (usually beneficial)
        for param in self.model.LayerNorm.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
