import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pdb

class OfflearningDataset(Dataset):
    def __init__(self, txt_path, frames_per_group=5):
        """
        Args:
            txt_path (str): 训练数据路径
            frames_per_group (int): 每组帧的数量，用于Loss平均
        """
         # 存储参数
        self.frames_per_group = frames_per_group
        # 读txt文件
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        self.frame_sizes = []  # 帧大小
        self.loss_counts = []  # 丢包数
        self.rtts = []         # rtt值
        for line in lines:
            if line.strip():  
                parts = line.strip().split()
                if len(parts) >= 3:  # 没空数据，回头去掉
                    self.frame_sizes.append(float(parts[0]))
                    self.loss_counts.append(float(parts[1]))
                    self.rtts.append(float(parts[2]))

        # 转为NumPy数组
        self.frame_sizes = np.array(self.frame_sizes)
        self.loss_counts = np.array(self.loss_counts)
        self.rtts = np.array(self.rtts)

        
        total_frames = len(self.frame_sizes)
        self.num_groups = total_frames // frames_per_group
        if total_frames % frames_per_group != 0:
            # 确保能被整除，可能需要丢弃末尾的几帧
            self.num_groups += 1
            # 填充逻辑，应该不用这样整了吧，数据不会出现id问题而不对齐
            pad_size = self.num_groups * frames_per_group - total_frames
            self.frame_sizes = np.pad(self.frame_sizes, (0, pad_size), 'edge')
            self.loss_counts = np.pad(self.loss_counts, (0, pad_size), 'edge')
            self.rtts = np.pad(self.rtts, (0, pad_size), 'edge')

            pdb.set_trace()#调试断点 

    def __len__(self):
        return self.num_groups

    def __getitem__(self, idx):
        #  起始和结束索引
        start_idx = idx * self.frames_per_group
        end_idx = start_idx + self.frames_per_group
        
        #  帧大小和丢包数
        group_frame_sizes = self.frame_sizes[start_idx:end_idx]
        group_loss_counts = self.loss_counts[start_idx:end_idx]
        group_rtt_counts = self.rtts[start_idx:end_idx]
        total_frames = len(group_frame_sizes)#总帧数量

        # 计算平均RTT，使用组内最后一帧的RTT作为代表
        total_rtt=np.sum(group_rtt_counts)
        avg_rtt = total_rtt / total_frames if total_frames > 0 else 0

        # 计算平均丢包率
        total_loss = np.sum(group_loss_counts)
        avg_loss_rate = total_loss / total_frames if total_frames > 0 else 0
        
        # 创建帧特征张量 - 只包含帧大小
        frame_samples = torch.tensor(group_frame_sizes, dtype=torch.float32).unsqueeze(1)
        
        # 创建丢包标志和计数
        loss_counts = torch.tensor(group_loss_counts, dtype=torch.float32)
        loss_flags = (loss_counts != 0)
        
        return {
            "frame_samples": frame_samples,
            "loss_flags": loss_flags,
            "loss_counts": loss_counts,
            "rtt": torch.tensor([avg_rtt], dtype=torch.float32),
            "avg_loss_rate": torch.tensor([avg_loss_rate], dtype=torch.float32),
        }

def collate_fn(batch):
    frame_samples = [item["frame_samples"] for item in batch]
    loss_flags = [item["loss_flags"] for item in batch]
    loss_counts = [item["loss_counts"] for item in batch]
    rtts = [item["rtt"] for item in batch]
    avg_loss_rates = [item["avg_loss_rate"] for item in batch]
    
    # 对变长序列进行padding,这里也不用了吧
    padded_frame_samples = pad_sequence(frame_samples, batch_first=True, padding_value=0.0)
    padded_loss_flags = pad_sequence(loss_flags, batch_first=True, padding_value=0.0)
    padded_loss_counts = pad_sequence(loss_counts, batch_first=True, padding_value=0.0)
    
    # 计算序列长度和掩码
    seq_lengths = torch.tensor([len(s) for s in frame_samples], dtype=torch.long)
    max_len = padded_frame_samples.shape[1]
    mask = torch.arange(max_len).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)
    
    return {
        "frame_samples": padded_frame_samples,
        "loss_flags": padded_loss_flags,
        "loss_counts": padded_loss_counts,
        "rtt": torch.stack(rtts),
        "avg_loss_rate": torch.stack(avg_loss_rates),
        "seq_lengths": seq_lengths,
        "mask": mask
    }