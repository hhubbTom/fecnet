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
        """
        self.df = pd.read_table(csv_path, sep='\s+', header=None) 
        self.chunk_size = 10
        self.num_chunks = len(self.df) // self.chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        chunk = self.df.iloc[start:end]

        # 计算 loss 和 rtt 的平均值
        mean_loss = chunk[1].mean()  # 对第1列（loss）取均值
        mean_rtt = chunk[2].mean()   # 对第2列（rtt）取均值
        
        return {
            "frames": chunk[0],
            "loss": mean_loss,
            "rtt":  mean_rtt
        }#返回字典
