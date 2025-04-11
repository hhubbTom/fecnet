import torch
import math
import torch.nn as nn
#处理变长序列，sin位置编码
class FrameTransformer(nn.Module):
    """ frame feature extractor

    Args:
        d_model: 64
        nhead: 2
        num_layers: 2 
        dim_feedforward: 128
    """
    def __init__(self, d_model=64, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        # Ascending dimension
        self.d_model = d_model

        self.embed = nn.Linear(1, d_model)  
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )#创建了Transformer编码器
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # transformer layer
        
        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),#激活函数，引入非线性
            nn.Linear(32, 16)
        )
    def generate_positional_encoding(self, seq_len, device):
        """ Dynamic generation of sin position encoding

        Args:
            seq_len: the max length of the batch
            device: cuda or cpu 

        Returns:
           pe: position encoding result. pe.size(): (batch, seq_len, d_model)
        """
        #创建一个序列位置索引，从0到seq_len-1。
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() * 
                            (-math.log(10000.0) / (self.d_model // 2))
                            )#用于位置编码的频率项，使不同位置和维度有不同的编码值
        
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)  #绝对位置编码
        
        return pe  # return (1, seq_len, d_model)

    def forward(self, x):
        
        # x shape: (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len) -> (batch, seq_len, 1)
        
        x = self.embed(x)  # 将1维特征扩展到64维,(batch, seq_len, 1) -> (batch, seq_len, d_model)
        
        # position encoding
        seq_len = x.size(1)
        pos_embed = self.generate_positional_encoding(seq_len, x.device)
        x += pos_embed # (batch, seq_len, d_model)
        
        # transformer layer将数据通过Transformer编码器处理
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # mean pool
        x = x.mean(dim=1)  # 对序列维度取平均，将所有时间步的信息压缩成一个向量(batch, seq_len, d_model)->(batch, d_model)
        return self.output(x)   #将64维特征映射到16维输出