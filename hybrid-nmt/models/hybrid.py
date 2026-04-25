"""
Hybrid Translation Model: IndicBERT Encoder + Projection Bridge + IndicTrans2 Decoder
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .encoder import IndicBERTEncoder
from .projection import ProjectionBridge
from .decoder import IndicTrans2Decoder


class HybridTranslationModel(nn.Module):
    """
    Hybrid neural machine translation model combining:
    - IndicBERT v2 as encoder (768-dim output)
    - ProjectionBridge for dimension mismatch (768 → 1024)
    - IndicTrans2 decoder (1024-dim input)
    
    Supports 3-phase training:
      Phase 1: Train only bridge (frozen encoder and decoder)
      Phase 2: Train bridge + top 4 decoder layers
      Phase 3: Train bridge + top 4 decoder + top 4 encoder layers
    """
    
    def __init__(
        self,
        encoder_model="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
        decoder_model="ai4bharat/indictrans2-en-indic-dist-200M",
        encoder_dim=768,
        decoder_dim=1024,
        dropout=0.1
    ):
        """
        Args:
            encoder_model (str): HuggingFace encoder model identifier
            decoder_model (str): HuggingFace decoder model identifier
            encoder_dim (int): Output dimension of encoder
            decoder_dim (int): Input dimension of decoder
            dropout (float): Dropout rate for projection bridge
        """
        super().__init__()
        
        # Initialize components
        self.encoder = IndicBERTEncoder(encoder_model)
        self.bridge = ProjectionBridge(encoder_dim, decoder_dim, dropout)
        self.decoder = IndicTrans2Decoder(decoder_model)
        
        # Training phase tracking
        self.current_phase = 1
        
        # Tokenizers
        self.src_tokenizer = AutoTokenizer.from_pretrained(encoder_model, trust_remote_code=True)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(decoder_model, trust_remote_code=True)
        
        # Padding token IDs for loss masking
        self.pad_token_id = self.tgt_tokenizer.pad_token_id
        
        print(f"[HybridTranslationModel] Architecture:")
        print(f"  Encoder (IndicBERT): {self.encoder.output_dim} dim")
        print(f"  Bridge: {encoder_dim} → {decoder_dim} dim")
        print(f"  Decoder (IndicTrans2): vocab_size={self.decoder.vocab_size}")
        print(f"  Pad token ID: {self.pad_token_id}")
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        """
        Forward pass through the full hybrid model.
        
        Args:
            src_input_ids (torch.LongTensor): [batch_size, src_seq_len]
            src_attention_mask (torch.LongTensor): [batch_size, src_seq_len]
            tgt_input_ids (torch.LongTensor): [batch_size, tgt_seq_len]
            tgt_attention_mask (torch.LongTensor): [batch_size, tgt_seq_len]
        
        Returns:
            torch.FloatTensor: [batch_size, tgt_seq_len, vocab_size]
                Logits over vocabulary
        """
        # Encoder pass
        encoder_hidden_states = self.encoder(src_input_ids, src_attention_mask)
        # [batch_size, src_seq_len, 768]
        
        # Bridge pass
        encoder_hidden_states = self.bridge(encoder_hidden_states)
        # [batch_size, src_seq_len, 1024]
        
        # Decoder pass
        logits = self.decoder(
            tgt_input_ids=tgt_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=src_attention_mask
        )
        # [batch_size, tgt_seq_len, vocab_size]
        
        return logits
    
    def set_phase(self, phase: int):
        """
        Set training phase and adjust frozen/unfrozen layers accordingly.
        
        Args:
            phase (int): Phase number (1, 2, or 3)
                Phase 1: Freeze encoder and decoder, train only bridge
                Phase 2: Freeze encoder, train bridge + top 4 decoder layers
                Phase 3: Train bridge + top 4 decoder + top 4 encoder layers
        """
        self.current_phase = phase
        
        if phase == 1:
            print("[PHASE 1] Training: Bridge only")
            print("  Frozen: Encoder, Decoder")
            print("  Training: Bridge")
            
            self.encoder.freeze()
            self.decoder.freeze()
            # Bridge is already unfrozen by default
        
        elif phase == 2:
            print("[PHASE 2] Training: Bridge + Top 4 Decoder Layers")
            print("  Frozen: Encoder")
            print("  Training: Bridge, Top 4 decoder layers")
            
            self.encoder.freeze()
            self.decoder.unfreeze_top_layers(n=4)
        
        elif phase == 3:
            print("[PHASE 3] Training: Bridge + Top 4 Decoder + Top 4 Encoder Layers")
            print("  Frozen: Bottom encoder layers")
            print("  Training: Bridge, Top 4 encoder layers, Top 4 decoder layers")
            
            self.encoder.unfreeze_top_layers(n=4)
            self.decoder.unfreeze_top_layers(n=4)
        
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")
        
        # Print trainable parameters
        enc_params = self.encoder.get_trainable_params()
        bridge_params = self.bridge.get_trainable_params()
        dec_params = self.decoder.get_trainable_params()
        total_params = enc_params + bridge_params + dec_params
        
        print(f"  Trainable params: Encoder={enc_params}, Bridge={bridge_params}, "
              f"Decoder={dec_params} (Total={total_params})")
    
    def generate(self, src_input_ids, src_attention_mask, max_length=128):
        """
        Greedy decoding for inference.
        
        Args:
            src_input_ids (torch.LongTensor): [batch_size, src_seq_len]
            src_attention_mask (torch.LongTensor): [batch_size, src_seq_len]
            max_length (int): Maximum generation length
        
        Returns:
            torch.LongTensor: [batch_size, max_length]
                Generated token IDs
        """
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        
        # Encode source
        encoder_hidden_states = self.encoder(src_input_ids, src_attention_mask)
        encoder_hidden_states = self.bridge(encoder_hidden_states)
        
        # Start with BOS token
        # For IndicTrans2, BOS is typically the language token
        eos_token_id = self.tgt_tokenizer.eos_token_id
        bos_token_id = self.tgt_tokenizer.bos_token_id
        
        current_tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Greedy generation
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.decoder(
                tgt_input_ids=current_tokens,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=src_attention_mask
            )
            # [batch_size, current_len, vocab_size]
            
            # Get best token (greedy)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            # [batch_size, 1]
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_tokens], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_tokens == eos_token_id).all():
                break
        
        return current_tokens
    
    def beam_search_generate(
        self,
        src_input_ids,
        src_attention_mask,
        beam_size=5,
        max_length=128,
        length_penalty=1.0
    ):
        """
        Beam search decoding for inference.
        
        Args:
            src_input_ids (torch.LongTensor): [batch_size, src_seq_len]
            src_attention_mask (torch.LongTensor): [batch_size, src_seq_len]
            beam_size (int): Beam width
            max_length (int): Maximum generation length
            length_penalty (float): Length penalty for beam search
        
        Returns:
            tuple: (generated_ids, scores)
                generated_ids: [batch_size, beam_size, max_length]
                scores: [batch_size, beam_size]
        """
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        
        # Encode source
        encoder_hidden_states = self.encoder(src_input_ids, src_attention_mask)
        encoder_hidden_states = self.bridge(encoder_hidden_states)
        
        # Expand for beam search
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(beam_size, dim=0)
        src_attention_mask = src_attention_mask.repeat_interleave(beam_size, dim=0)
        
        eos_token_id = self.tgt_tokenizer.eos_token_id
        bos_token_id = self.tgt_tokenizer.bos_token_id
        
        # Initialize beams
        current_tokens = torch.full((batch_size * beam_size, 1), bos_token_id, dtype=torch.long, device=device)
        scores = torch.zeros((batch_size * beam_size, 1), device=device)
        finished = torch.zeros((batch_size * beam_size,), dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            # Get logits
            logits = self.decoder(
                tgt_input_ids=current_tokens,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=src_attention_mask
            )
            
            # Get log probabilities
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            # [batch_size * beam_size, vocab_size]
            
            # Add previous scores (normalized by beam)
            # For the first step, all beams have same score
            if step == 0:
                log_probs = log_probs[:batch_size]  # Use only first beam entries
                top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                # [batch_size, beam_size]
            else:
                # Score: previous_score + log_prob
                scores_expanded = scores.view(batch_size, beam_size)
                top_log_probs, top_indices = torch.topk(
                    log_probs + scores_expanded.view(-1, 1),
                    beam_size,
                    dim=-1
                )
            
            # Flatten and get corresponding tokens
            top_tokens = top_indices.view(-1)
            new_scores = top_log_probs.view(-1, 1) / ((step + 2) ** length_penalty)
            
            # Reshape sequences
            if step == 0:
                # First step: replicate sequences
                current_tokens = current_tokens.repeat(beam_size, 1)
            
            # Append tokens
            current_tokens = torch.cat([current_tokens, top_tokens.unsqueeze(-1)], dim=1)
            scores = new_scores
        
        return current_tokens, scores
    
    def get_trainable_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
