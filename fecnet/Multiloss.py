import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
#损失函数设计
# 1. Fec优化损失，这个loss怎么改？？

class OfflearningLoss(nn.Module):#离线的
    def __init__(self, fec_bins):
        """
        Args:
            fec_bins: Predefined fec table boundaries
        """
        super().__init__()
        self.num_bins = len(fec_bins) 
        self.register_buffer('fec_bins', fec_bins.float())#注册一个不需要梯度的张量缓冲区，存储FEC分段边界

    def forward(self, fec_table, 
                frame_samples, loss):
        """
        Args:
            fec_table: 预测的FEC表
            frame_samples: 帧大小
            loss:实际标签
        """

        mask = loss>0 #mask和loss都是Size([32, 10])
        if not mask.any():
        # 如果没有丢包，返回零损失
            return torch.tensor(0.0, device=fec_table.device, requires_grad=True)
        #pdb.set_trace()
        #frame_samples.shape Size([32, 10, 1])
        frame_sizes = frame_samples[mask].float().squeeze(-1)   #目前是([75, 1]),18下午变[24, 1]了
        #loss_counts = loss_counts[mask].float()      
        
        #actual_loss_rate = loss_counts / frame_sizes
        # 将帧大小映射到FEC表中对应的索引

        """
            frame_samples = torch.tensor([100, 30, 5, 200])  # 4个帧的大小
            loss = torch.tensor([10, 0, 1, 20])              # 对应的丢包数量
            掩码结果: tensor([True, False, True, True])

            frame_sizes = frame_samples[mask]
            # 结果: tensor([100, 5, 200])
            loss_counts = loss[mask]
            # 结果: tensor([10, 1, 20])

            batch_indices = torch.nonzero(mask, as_tuple=True)[0]
            # 结果: tensor([0, 2, 3])  # 第0、2、3个样本有丢包
            bin_indices = torch.bucketize(frame_sizes, fec_bins, right=True)
            # 结果: tensor([9, 2, 10])
            
            查表即可,
            predicted_fec = [
        fec_table[0, 9],  # 样本0的第9个桶的FEC值
        fec_table[2, 2],  # 样本2的第2个桶的FEC值
        fec_table[3, 10]  # 样本3的第10个桶的FEC值 (可能越界)
        ]
        """

        bin_indices = torch.bucketize(frame_sizes, self.fec_bins, right=True)  # [75, 1] [24,1]
        bin_indices = torch.clamp(bin_indices, min=0, max=self.num_bins-1)
        #bin_indices = bin_indices.squeeze(-1)
        real_loss = loss[mask].float()
        batch_indices = torch.nonzero(mask, as_tuple=True)[0] #表示哪些位置loss>0,.Size([75]) [24]

        predicted_fec = fec_table[batch_indices,bin_indices].squeeze(-1) #计算实际丢包率，并根据帧大小查表 fec_table是[batch,10]
        predicted_fec_counts = predicted_fec * frame_sizes
        # 计算实际丢包率
        actual_loss_rate = real_loss / frame_sizes

        under_protect = predicted_fec_counts < real_loss
        over_protect = ~under_protect
        # 偏置系数：鼓励FEC略大于实际丢包数
        safety_margin = 1.5  # 希望FEC至少比实际丢包数高出10%
        optimal_fec = real_loss * safety_margin
        # 计算带有不对称权重的损失
        under_weight = 12.0  # 欠保护的惩罚权重
        over_weight = 0.1   # 过保护的惩罚权重
        

        # 使用带有安全边际的MSE损失
        if under_protect.any():
            loss_under = ((optimal_fec[under_protect] - predicted_fec_counts[under_protect]) ** 2).sum() * under_weight
        else:
            loss_under = 0.0
            
        if over_protect.any():
            # 对于过保护情况，我们只惩罚超过安全边际的部分
            over_margin = predicted_fec_counts[over_protect] > optimal_fec[over_protect]
            if over_margin.any():
                excess = predicted_fec_counts[over_protect][over_margin] - optimal_fec[over_protect][over_margin]
                loss_over = (excess ** 2).sum() * over_weight
            else:
                loss_over = 0.0
        else:
            loss_over = 0.0
        #pdb.set_trace()
        # 计算总损失
        fec_loss = (loss_under + loss_over) / max(len(actual_loss_rate), 1)
        
        return fec_loss