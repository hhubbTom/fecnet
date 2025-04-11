# config.py
import torch
import os

class Config:
    """config"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = "frame_loss_rtt.txt"
        self.frames_per_group = 5  # 用于Loss平均，防止拟合？
        # FEC预测的分级bins
        self.fec_bins = torch.tensor([ 1, 2, 3, 4, 6, 8, 10, 12, 14, 20, 30, 45, 60, 80, 100])
        # 帧Transformer模型参数
        self.frame_transformer_params = {
            "d_model": 64,      # 模型维度
            "nhead": 2,         # 多头注意力机制的头数
            "num_layers": 2,    # Transformer层数
            "dim_feedforward": 128  # 前馈网络隐藏层维度
        }
        # FECNet模型参数
        self.fecnet_params = {
            "input_dim": 2  # 改为2，只有loss_rate和rtt
        }

        self.num_epochs = 20
        self.batch_size = 32
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.resume_checkpoint = None
        self.optimizer_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0   # 要增加权重衰减吗？
        }
        self.scheduler_params = {
            "step_size": 5,
            "gamma": 0.1
        }
        # 损失函数权重
        self.loss_weights = {
            "bitrate": 1.0,
            "fec": 2.0
        }
config = Config()
