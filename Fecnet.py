import torch
import torch.nn as nn
#全连接层，处理loss，fc，delay_based GCC的输出值，融合特征，再通过两个预测头回归出bitrate和fec。
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
            nn.LayerNorm(8)
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
        )
        
        merged_dim = 16 + 8 * 3
        
        self.joint_processor = nn.Sequential(
            nn.Linear(merged_dim, 64),  
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.bitrate_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.fec_head = nn.Sequential(
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
        merged = self.joint_processor(merged)
        bitrate = self.bitrate_head(merged).squeeze(-1)  # (batch,)
        fec_ratios = self.fec_head(merged)               # (batch, 10)
        
        return bitrate, fec_ratios


