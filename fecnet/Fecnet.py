import torch
import torch.nn as nn

#全连接层，处理loss，fc，delay_based GCC的输出值，融合特征，再通过两个预测头回归出bitrate和fec。

# 多任务学习：通过共享特征表示学习多个相关任务，可以提高每个任务的性能。
# 特征融合：将来自不同源的特征有效地组合起来，形成更丰富的表示。
# 输出头：在共享特征表示之上，为每个任务添加专门的预测层

class fecnet(nn.Module):
    """network for fec generation
    
    Args:
        frame_transformer: 
        rtt_input_dim: 未知
    """
    def __init__(self, frame_transformer):
        super().__init__()
        self.frame_transformer = frame_transformer
        
        self.loss_fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)# 归一化层
        )
        self.rtt_fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )
        
        merged_dim = 16 + 8 * 2     #计算合并后的特征维度，去掉了gcc的8
        
        self.joint_processor = nn.Sequential(  #处理特征
            nn.Linear(merged_dim, 64),  
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.fec_head = nn.Sequential(  #FEC预测头，输出15个
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 15),
            nn.Sigmoid()  # 0-1
        )

    def forward(self, frame_seq, loss_rate, rtt, gcc_features):
        """forward

        Args:
            frame_seq: 
            loss_rate: 
            rtt: 

        Returns:
            fec_ratio:[batch, 15]
        """
        #transformers的输入是一个batch的序列数据，输出是一个batch的特征表示？
        # frame_seq shape: (batch, 16)
        frame_feat = self.frame_transformer(frame_seq)  # (batch, 16)
        
        loss_feat = self.loss_fc(loss_rate.unsqueeze(-1))  # (batch, 8)
        rtt_feat = self.rtt_fc(rtt.unsqueeze(-1))          # (batch, 8)

        merged = torch.cat([frame_feat, loss_feat, rtt_feat], dim=1)
        merged = self.joint_processor(merged)#处理联合特征
        
        fec_ratios = self.fec_head(merged)               # (batch, 15)
        
        return fec_ratios


