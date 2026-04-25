"""
IndicTrans2 Decoder wrapper (decoder only, no encoder)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class IndicTrans2Decoder(nn.Module):
    """
    IndicTrans2 decoder for the hybrid NMT model.
    
    Extracts ONLY the decoder and lm_head from the full IndicTrans2 model.
    The encoder from IndicTrans2 is discarded entirely.
    
    Input dimension: 1024 (from ProjectionBridge)
    Output: logits over vocabulary
    """
    
    def __init__(self, model_name="ai4bharat/indictrans2-en-indic-dist-200M"):
        """
        Args:
            model_name (str): HuggingFace model identifier
        """
        super().__init__()
        
        # Load full model
        full_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Extract ONLY decoder and lm_head
        # IndicTrans2 uses bart-style architecture: model.model.decoder
        self.decoder = full_model.model.decoder
        self.lm_head = full_model.lm_head
        
        # Store config
        self.config = full_model.config
        self.vocab_size = full_model.config.vocab_size
        self.output_dim = self.vocab_size
        
        # Get tokenizer for BOS token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Note: We discard full_model and its encoder entirely
        del full_model
    
    def forward(self, tgt_input_ids, encoder_hidden_states, encoder_attention_mask):
        """
        Forward pass through the decoder.
        
        Args:
            tgt_input_ids (torch.LongTensor): [batch_size, tgt_seq_len]
                Target input token IDs
            encoder_hidden_states (torch.FloatTensor): [batch_size, src_seq_len, 1024]
                Projected encoder output from the bridge
            encoder_attention_mask (torch.LongTensor): [batch_size, src_seq_len]
                Attention mask for encoder (source) sequence
        
        Returns:
            torch.FloatTensor: [batch_size, tgt_seq_len, vocab_size]
                Logits over vocabulary
        """
        # Get decoder embeddings
        decoder_outputs = self.decoder(
            input_ids=tgt_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        # Get last hidden state
        last_hidden_state = decoder_outputs.last_hidden_state  # [batch, tgt_seq_len, decoder_dim]
        
        # Project to vocabulary
        logits = self.lm_head(last_hidden_state)  # [batch, tgt_seq_len, vocab_size]
        
        return logits
    
    def freeze(self):
        """Freeze all decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, n=4):
        """
        Unfreeze the top N decoder layers.
        
        Args:
            n (int): Number of top layers to unfreeze
        """
        # Freeze everything first
        self.freeze()
        
        # Unfreeze top N layers
        num_layers = len(self.decoder.layers)
        
        for i in range(num_layers - n, num_layers):
            for param in self.decoder.layers[i].parameters():
                param.requires_grad = True
        
        # Unfreeze layer norm and projections
        for param in self.decoder.layernorm_embedding.parameters():
            param.requires_grad = True
        
        # Unfreeze lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Count trainable parameters"""
        total = 0
        total += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        return total
