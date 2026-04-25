"""
PyTorch Dataset class for loading tokenized translation pairs
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation pairs.
    
    Loads tokenized pairs and returns:
        - src_input_ids (torch.LongTensor)
        - src_attention_mask (torch.LongTensor)
        - tgt_input_ids (torch.LongTensor)
        - tgt_attention_mask (torch.LongTensor)
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path (str or Path): Path to .pt file containing tokenized pairs
        """
        self.data_path = Path(data_path)
        self.data = torch.load(self.data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized pair as a dictionary of tensors
        """
        sample = self.data[idx]
        
        return {
            "src_input_ids": sample["src_input_ids"],
            "src_attention_mask": sample["src_attention_mask"],
            "tgt_input_ids": sample["tgt_input_ids"],
            "tgt_attention_mask": sample["tgt_attention_mask"],
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Stacks individual samples into batches.
    
    Args:
        batch (list): List of samples from __getitem__
    
    Returns:
        dict: Batched tensors
    """
    return {
        "src_input_ids": torch.stack([sample["src_input_ids"] for sample in batch]),
        "src_attention_mask": torch.stack([sample["src_attention_mask"] for sample in batch]),
        "tgt_input_ids": torch.stack([sample["tgt_input_ids"] for sample in batch]),
        "tgt_attention_mask": torch.stack([sample["tgt_attention_mask"] for sample in batch]),
    }
