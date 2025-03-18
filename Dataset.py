import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class OfflearningDataset(Dataset):
    def __init__(self, data_dir, fec_bins):
        """
        Args:
            data_dir (str): dataset
            fec_bins (torch.Tensor): FEC frame level
        """
        self.data_dir = data_dir
        self.fec_bins = fec_bins
        self.batch_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".csv")],
            key=lambda x: int(x.split(".")[0])
        )

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.batch_files[idx])
        df = pd.read_csv(file_path, header=None)
        
        gcc_bitrate = float(df.iloc[:, 4])  # GCC
        rtt = float(df.iloc[:, 5])          # RTT

        # 解析帧数据（跳过前两行）
        frame_data = df.iloc[2:, :4].values  # 前四列是帧数据
        frame_samples = torch.tensor(frame_data[:, :2], dtype=torch.float32)
        loss_flags = torch.tensor(frame_data[:, 2], dtype=torch.float32)
        loss_counts = torch.tensor(frame_data[:, 3], dtype=torch.float32)

        return {
            "gcc_bitrate": torch.tensor([gcc_bitrate]),
            "rtt": torch.tensor([rtt]),
            "frame_samples": frame_samples,
            "loss_flags": loss_flags,
            "loss_counts": loss_counts
        }