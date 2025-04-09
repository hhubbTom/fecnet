import torch
import torch.nn as nn
# 基础概念
# 多任务学习：通过共享特征表示学习多个相关任务，可以提高每个任务的性能。
# 特征融合：将来自不同源的特征有效地组合起来，形成更丰富的表示。
# 输出头：在共享特征表示之上，为每个任务添加专门的预测层
class fecnet(nn.Module):
    """network for bitrate and fec generation
    
    Args:
        frame_transformer: 
        gcc_input_dim: unknown dim
    """
    def __init__(self, frame_transformer, gcc_input_dim=5):
        super().__init__()
        self.frame_transformer = frame_transformer
        
        self.loss_fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)#LayerNorm是一种归一化层
        )
        self.rtt_fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )

        self.gcc_fc = nn.Sequential(
            nn.Linear(gcc_input_dim, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )#创建处理丢包率,处理RTT和GCC特征的三个全连接网络
        
        merged_dim = 16 + 8 * 3     #计算合并后的特征维度
        
        self.joint_processor = nn.Sequential(  #处理合并后的特征
            nn.Linear(merged_dim, 64),  
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.bitrate_head = nn.Sequential(  #创建比特率预测头，将联合特征映射到单个比特率值
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.fec_head = nn.Sequential(  #创建FEC预测头，输出10个FEC比率，值范围在0-1之间
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()  # 0-1
        )

    def forward(self, frame_seq, loss_rate, rtt, gcc_features):
        """forward

        Args:
            frame_seq: 
            loss_rate: 
            rtt: 
            gcc_features: 

        Returns:
            bitrate, fec_ratio
        """

        frame_feat = self.frame_transformer(frame_seq)  # (batch, 16)
        
        loss_feat = self.loss_fc(loss_rate.unsqueeze(-1))  # (batch, 8)
        rtt_feat = self.rtt_fc(rtt.unsqueeze(-1))          # (batch, 8)
        gcc_feat = self.gcc_fc(gcc_features)               # (batch, 8)

        merged = torch.cat([frame_feat, loss_feat, rtt_feat, gcc_feat], dim=1)
        merged = self.joint_processor(merged)#通过联合处理器进一步处理联合特征
        bitrate = self.bitrate_head(merged).squeeze(-1)  # (batch,)
        fec_ratios = self.fec_head(merged)               # (batch, 10)
        
        return bitrate, fec_ratios


