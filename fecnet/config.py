# config.py
import torch

class Config:
    """config"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = "data_newgs.csv"
        # FEC预测的分级bins，这些值代表不同的冗余比率
        self.fec_bins = torch.tensor([2, 4, 6, 8, 10, 12, 14, 20, 50, 100])
        # 帧Transformer模型参数
        self.frame_transformer_params = {
            "d_model": 64,      # 模型维度
            "nhead": 2,         # 多头注意力机制的头数
            "num_layers": 2,    # Transformer层数
            "dim_feedforward": 128  # 前馈网络隐藏层维度
        }
        # FECNet模型参数-移除了gcc_input_dim参数

        self.num_epochs = 20
        self.batch_size = 32
        self.checkpoint_dir = "checkpoints"
        self.resume_checkpoint = None  # 例如："checkpoints/checkpoint_epoch_10.pt"
        self.optimizer_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0
        }
        self.scheduler_params = {
            "step_size": 5,
            "gamma": 0.1
        }
config = Config()
