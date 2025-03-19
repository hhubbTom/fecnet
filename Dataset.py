import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import pdb

class OfflearningDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (str): train data
            fec_bins (torch.Tensor): FEC frame level
        """
        self.df = pd.read_csv(csv_path, header=None)
        self.group_keys = []
        self.group_indices = []
        current_group = []
        prev_session = None

        for idx, row in self.df.iterrows():
            session = row[0]  
            if session != prev_session and len(current_group) > 0:
                self.group_indices.append(current_group)
                self.group_keys.append(prev_session)
                current_group = []
            current_group.append(idx)
            prev_session = session
            
        if len(current_group) > 0:
            self.group_indices.append(current_group)
            self.group_keys.append(prev_session)
            pdb.set_trace()

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        group_rows = self.df.iloc[self.group_indices[idx]]
        gcc_bw = group_rows.iloc[0, 5]  
        rtt = group_rows.iloc[0, 6]     
        frame_data = group_rows.iloc[:, :5].values 
        frame_samples = torch.tensor(frame_data[:, [0, 2, 3, 4]], dtype=torch.float32)
        loss_counts = torch.tensor(frame_data[:, 1], dtype=torch.float32)
        loss_flags = (loss_counts != 0)
        
        return {
            "gcc_bw": torch.tensor([gcc_bw], dtype=torch.float32),
            "rtt": torch.tensor([rtt], dtype=torch.float32),
            "frame_samples": frame_samples,
            "loss_flags": loss_flags,
            "loss_counts": loss_counts
        }

def collate_fn(batch):
    gcc_bws = [item["gcc_bw"] for item in batch]
    rtts = [item["rtt"] for item in batch]
    frame_samples = [item["frame_samples"] for item in batch]
    loss_flags = [item["loss_flags"] for item in batch]
    loss_counts = [item["loss_counts"] for item in batch]
    
    # 对变长序列进行padding
    padded_frame_samples = pad_sequence(
        frame_samples, 
        batch_first=True, 
        padding_value=0.0
    )
    
    padded_loss_flags = pad_sequence(
        loss_flags,
        batch_first=True,
        padding_value=0.0
    )
    
    padded_loss_counts = pad_sequence(
        loss_counts,
        batch_first=True,
        padding_value=0.0
    )
    
    seq_lengths = torch.tensor([len(s) for s in frame_samples], dtype=torch.long)
    max_len = padded_frame_samples.shape[1]
    mask = torch.arange(max_len).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)
    
    return {
        "gcc_bw": torch.stack(gcc_bws),
        "rtt": torch.stack(rtts),
        "frame_samples": padded_frame_samples,
        "loss_flags": padded_loss_flags,
        "loss_counts": padded_loss_counts,
        "seq_lengths": seq_lengths,
        "mask": mask
    }