import torch
import torch.nn as nn
import torch.nn.functional as F
#损失函数设计
# 1. Fec优化损失，这个loss怎么改？？

class Multiloss(nn.Module):
    """The loss function of three optimization objectives
    
    Args:
        packet_size:
        alpha:
        beta: 
        gamma: 
        device:

    """
    def __init__(self, packet_size=1200, alpha=1.0, beta=3.0, gamma=0.5, device='cuda'):
        super().__init__()
        self.packet_size = packet_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  #alpha, beta, gamma: 不同损失项的权重系数,gamma暂时没用上??
        self.device = device

    def forward(self, pred_bitrate, pred_fec, fec_level_table, frame_result):
        """
        Args:
            pred_bitrate: 
            pred_fec:     
            frame_result:   帧处理结果
            fec_level:  FEC级别表
        """

        
        # to tensor  ----将输入的帧结果转换为张量
        frame_tensor_dict = {}
        for key in ['frame_size', 'fec', 'loss_packets', 'recovery_status']:
           sample_tensors = [torch.tensor(sample) for sample in frame_result[key]]
           frame_tensor_dict[key] = sample_tensors.to(self.device)
        
        frame_size = frame_tensor_dict['frame_size']
        loss_packets = frame_tensor_dict['loss_packets']
        recovery_status = frame_tensor_dict['recovery_status'] # 1 if the recovery is successful, 0 else. 
        
        # calculate fec value based on frame_size
        # actually, level table can be learned,这个表可以学习
        #根据帧大小和FEC级别表确定FEC比率和FEC包数量
        indices = torch.searchsorted(fec_level_table, frame_size, side='left') - 1
        fec_ratio = pred_fec[indices]
        fec_packets_num = fec_ratio * frame_size
        
        #--------------------Fec optimization loss-------------------#
        # Calculate the fec packet required for each frame based on the frame size
        
        diff = loss_packets - fec_packets_num
        l2_norm = torch.norm(diff, p=2)
        loss_fec_opt = self.alpha * l2_norm[recovery_status]  + self.beta * l2_norm[torch.logical_not(recovery_status)] # The first item is recovery success item
        

        
        return loss_fec_opt
    

class OfflearningLoss(nn.Module):#离线的
    def __init__(self, fec_bins):
        """
        Args:
            fec_bins: Predefined fec table boundaries
        """
        super().__init__()
        self.register_buffer('fec_bins', fec_bins.float())#注册一个不需要梯度的张量缓冲区，存储FEC分段边界

    def forward(self, pred_bitrate, gcc_bitrate, fec_table, 
                frame_samples, loss_flags, loss_counts, delay_gradient):
        """
        Args:
            pred_bitrate: 预测的比特率
            fec_table: 预测的FEC表
            frame_samples: 帧大小
            loss_flags: 丢包标志
            loss_counts: 丢包计数
            rtt: 往返时延
        """
        #下面6行,计算比特率损失，考虑了过预测和欠预测的不同影响，通过延迟梯度调整权重。这是一种非对称损失，在网络拥塞时更严格地惩罚过高的比特率预测
        # bitrate_loss = F.mse_loss(pred_bitrate, gcc_bitrate)
        bitrate_diff = pred_bitrate - gcc_bitrate
        positive_diff = torch.clamp(bitrate_diff, min=0)  
        negative_diff = torch.clamp(-bitrate_diff, min=0) 
        
        over_weight = delay_gradient  # 假设gcc_congestion已归一化到[0,1]
        under_weight = 1 - delay_gradient        
        bitrate_loss = (positive_diff**2 * over_weight + negative_diff**2 * under_weight).mean()
        
        mask = loss_flags.bool()
        if not mask.any():#如果没有丢包，只返回比特率损失
            return bitrate_loss  # no packets loss
        frame_sizes = frame_samples[mask].float()   
        loss_counts = loss_counts[mask].float()      
        
        actual_loss_rate = loss_counts / frame_sizes
        bin_indices = torch.bucketize(frame_sizes, self.fec_bins, right=True) # (a, b]
        batch_indices = torch.arange(fec_table.size(0), device=fec_table.device)[:, None]
        predicted_fec = fec_table[batch_indices,bin_indices] #计算实际丢包率，并根据帧大小查表
        
        under_protect = actual_loss_rate > predicted_fec
        over_protect = actual_loss_rate <= predicted_fec

        loss_under = (actual_loss_rate[under_protect] - predicted_fec[under_protect]).sum() * 3
        loss_over = (predicted_fec[over_protect] - actual_loss_rate[over_protect]).sum()
        fec_loss = (loss_under + loss_over) / max(len(actual_loss_rate), 1)
        
        return bitrate_loss + fec_loss