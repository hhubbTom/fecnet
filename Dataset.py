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
        self.df = pd.read_csv(csv_path, header=None)#使用pandas读取CSV文件，不将第一行作为表头
        self.group_keys = []
        self.group_indices = []
        current_group = []
        prev_session = None

        for idx, row in self.df.iterrows():#遍历数据集中的每一行,当发现新的会话ID时，将前一个会话的所有索引保存起来
            session = row[0]  
            if session != prev_session and len(current_group) > 0:
                self.group_indices.append(current_group)
                self.group_keys.append(prev_session)
                current_group = []
            current_group.append(idx)
            prev_session = session
        del self.group_keys[0]#删除第一个组键,这是??
        if len(current_group) > 0:
            self.group_indices.append(current_group)
            self.group_keys.append(prev_session)

            int_list = [int(x) for x in self.group_keys]#计算会话ID的最小值和最大值，然后找出这个范围内缺失的ID。
            min_val  = min(int_list)
            max_val = max(int_list)
            full = set(range(min_val,max_val+1))
            miss = sorted(full-set(int_list))#
            pdb.set_trace()#调试断点,用于检查缺失的session id

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        group_rows = self.df.iloc[self.group_indices[idx]]
        gcc_bw = group_rows.iloc[0, 5]  
        rtt = group_rows.iloc[0, 6]     
        frame_data = group_rows.iloc[:, :5].values 
        frame_samples = torch.tensor(frame_data[:, [0, 2, 3, 4]], dtype=torch.float32)#选择列0,2,3,4
        loss_counts = torch.tensor(frame_data[:, 1], dtype=torch.float32)
        loss_flags = (loss_counts != 0)#从帧数据中提取丢包计数，并创建布尔丢包标志
        
        return {
            "gcc_bw": torch.tensor([gcc_bw], dtype=torch.float32),
            "rtt": torch.tensor([rtt], dtype=torch.float32),
            "frame_samples": frame_samples,
            "loss_flags": loss_flags,
            "loss_counts": loss_counts
        }#返回字典

def collate_fn(batch):#批处理函数
    gcc_bws = [item["gcc_bw"] for item in batch]
    rtts = [item["rtt"] for item in batch]
    frame_samples = [item["frame_samples"] for item in batch]
    loss_flags = [item["loss_flags"] for item in batch]
    loss_counts = [item["loss_counts"] for item in batch]#这5行是从批次中提取各个特征
    
    # 对变长序列进行padding-填充的意思
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
    #下面是创建掩码，标记哪些是真实数据，哪些是填充。
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