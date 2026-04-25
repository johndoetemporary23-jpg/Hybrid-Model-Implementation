"""
Comprehensive training script for Hybrid NMT model
- 3-phase training (bridge only → bridge+decoder → bridge+decoder+encoder)
- Mixed precision training with autocast and GradScaler
- Checkpointing and early stopping
- Validation with BLEU computation
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import HybridTranslationModel
from data.dataset import TranslationDataset, collate_fn

# ================================================================================
# CONFIGURATION & SETUP
# ================================================================================

CONFIG_PATH = "config.yaml"
DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract config values
ENCODER_MODEL = config["encoder_model"]
DECODER_MODEL = config["decoder_model"]
BATCH_SIZE = config["batch_size"]
LR_PHASE1 = config["learning_rate"]
LR_PHASE2 = config["learning_rate_phase2"]
LR_PHASE3 = config["learning_rate_phase3"]
WEIGHT_DECAY = config["weight_decay"]
WARMUP_STEPS = config["warmup_steps"]
LABEL_SMOOTHING = config["label_smoothing"]
MAX_GRAD_NORM = config["max_grad_norm"]
FP16 = config["fp16"]

NUM_EPOCHS = config["num_epochs"]
PHASE1_EPOCHS = config["phase1_epochs"]
PHASE2_EPOCHS = config["phase2_epochs"]
PHASE3_EPOCHS = config["phase3_epochs"]

EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
SRC_LANG = config["src_lang"]
TGT_LANG = config["tgt_lang"]

# Create directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# LOGGING & UTILITIES
# ================================================================================

class TrainingLogger:
    """Simple training logger"""
    
    def __init__(self, log_dir=RESULTS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.metrics = defaultdict(list)
    
    def log(self, epoch, phase, **kwargs):
        """Log metrics for an epoch"""
        entry = {
            "epoch": epoch,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.metrics[epoch] = entry
        
        # Print to console
        msg = f"[Epoch {epoch:2d} | Phase {phase}] "
        for key, val in kwargs.items():
            if isinstance(val, float):
                msg += f"{key}={val:.4f} "
            else:
                msg += f"{key}={val} "
        print(msg)
    
    def save(self):
        """Save metrics to JSON"""
        with open(self.log_file, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"\nLogging saved to {self.log_file}")

logger = TrainingLogger()

# ================================================================================
# TRAINING HELPERS
# ================================================================================

def load_data(split="train"):
    """Load dataset"""
    data_path = DATA_DIR / "splits" / f"{split}.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return TranslationDataset(data_path)

def create_dataloaders(batch_size=32, num_workers=4):
    """Create train, val, test dataloaders"""
    train_dataset = load_data("train")
    val_dataset = load_data("val")
    test_dataset = load_data("test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def create_optimizer_and_scheduler(model, lr, total_steps):
    """Create optimizer and learning rate scheduler"""
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Linear warmup + cosine annealing
    def lr_lambda(current_step: int):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch, phase):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/Phase {phase}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        src_input_ids = batch["src_input_ids"].to(DEVICE)
        src_attention_mask = batch["src_attention_mask"].to(DEVICE)
        tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)
        tgt_attention_mask = batch["tgt_attention_mask"].to(DEVICE)
        
        # Forward pass with mixed precision
        with amp.autocast(enabled=FP16):
            logits = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
            # [batch_size, tgt_seq_len, vocab_size]
            
            # Compute loss (ignore padding tokens)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_input_ids.view(-1)
            )
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": avg_loss:.4f})
    
    return total_loss / num_batches

def evaluate(model, val_loader, criterion, epoch, phase):
    """Validate model and compute metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}/Phase {phase}")
        
        for batch in pbar:
            src_input_ids = batch["src_input_ids"].to(DEVICE)
            src_attention_mask = batch["src_attention_mask"].to(DEVICE)
            tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)
            tgt_attention_mask = batch["tgt_attention_mask"].to(DEVICE)
            
            # Forward pass
            with amp.autocast(enabled=FP16):
                logits = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt_input_ids.view(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches:.4f})
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, epoch, phase, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    # Save as latest
    latest_path = CHECKPOINTS_DIR / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save every N epochs
    if epoch % config["save_every_n_epochs"] == 0:
        epoch_path = CHECKPOINTS_DIR / f"epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"  Saved checkpoint: {epoch_path}")
    
    # Save best model
    if is_best:
        best_path = CHECKPOINTS_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ Saved best model: {best_path}")

def load_checkpoint_if_exists(model, optimizer):
    """Load checkpoint if it exists"""
    latest_path = CHECKPOINTS_DIR / "latest.pt"
    if latest_path.exists():
        print(f"\nResuming from checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"] + 1, checkpoint["phase"]
    return 1, 1

# ================================================================================
# MAIN TRAINING LOOP
# ================================================================================

def main():
    import math
    
    print("=" * 80)
    print("HYBRID NMT TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Encoder: {ENCODER_MODEL}")
    print(f"  Decoder: {DECODER_MODEL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Mixed precision (FP16): {FP16}")
    print(f"  Device: {DEVICE}")
    print(f"\nTraining phases:")
    print(f"  Phase 1: Epochs 1-{PHASE1_EPOCHS} (Bridge only, LR={LR_PHASE1})")
    print(f"  Phase 2: Epochs {PHASE1_EPOCHS+1}-{PHASE1_EPOCHS+PHASE2_EPOCHS} (Bridge+Decoder, LR={LR_PHASE2})")
    print(f"  Phase 3: Epochs {PHASE1_EPOCHS+PHASE2_EPOCHS+1}-{NUM_EPOCHS} (Full model, LR={LR_PHASE3})")
    print()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print()
    
    # Create model
    print("Creating model...")
    model = HybridTranslationModel(ENCODER_MODEL, DECODER_MODEL)
    model = model.to(DEVICE)
    print(f"  Total trainable params: {model.get_trainable_params():,}")
    print()
    
    # Loss function (with label smoothing and padding ignore)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
        ignore_index=model.pad_token_id,
        reduction="mean"
    )
    
    # Gradient scaler for mixed precision
    scaler = amp.GradScaler(enabled=FP16)
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    phase_results = {}
    
    start_epoch, start_phase = 1, 1
    
    # Main training loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Determine current phase
        if epoch <= PHASE1_EPOCHS:
            phase = 1
            lr = LR_PHASE1
        elif epoch <= PHASE1_EPOCHS + PHASE2_EPOCHS:
            phase = 2
            lr = LR_PHASE2
        else:
            phase = 3
            lr = LR_PHASE3
        
        # Switch phase if needed
        if phase != start_phase or epoch == start_epoch:
            print(f"\n{'='*80}")
            model.set_phase(phase)
            
            # Create new optimizer for new phase
            total_steps = (NUM_EPOCHS - epoch + 1) * len(train_loader)
            optimizer, scheduler = create_optimizer_and_scheduler(model, lr, total_steps)
            start_phase = phase
            print(f"{'='*80}\n")
        
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch, phase)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, epoch, phase)
        
        # Log
        logger.log(epoch, phase, train_loss=train_loss, val_loss=val_loss)
        
        # Early stopping check
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, phase, is_best=is_best)
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break
        
        # Phase summary
        if epoch == PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS or epoch == NUM_EPOCHS:
            phase_results[phase] = {
                "best_val_loss": best_val_loss,
                "epochs": epoch
            }
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest model saved to: {CHECKPOINTS_DIR / 'best_model.pt'}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print()
    
    logger.save()

if __name__ == "__main__":
    main()
