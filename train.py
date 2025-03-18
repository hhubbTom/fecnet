import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from datetime import datetime
from FrameTransformer import FrameTransformer
from Fecnet import fecnet
from Multiloss import OfflearningLoss
from Dataset import OfflearningDataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# 定义训练流程
def train(
    data_dir, 
    fec_bins, 
    model, 
    loss_fn, 
    num_epochs=10, 
    lr=1e-3, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # 初始化 Dataset 和 DataLoader
    dataset = OfflearningDataset(data_dir, fec_bins)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 每个 CSV 文件是一个 batch

    # 模型和损失函数移至设备
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # 数据移至设备
            frame_samples = batch_data["frame_samples"].to(device)
            loss_flags = batch_data["loss_flags"].to(device)
            loss_counts = batch_data["loss_counts"].to(device)
            gcc_bitrate = batch_data["gcc_bitrate"].to(device)
            rtt = batch_data["rtt"].to(device)

            # ----------------------
            # 前向传播
            # ----------------------
            # 提取模型输入特征
            batch_size = 1  # 每个 CSV 对应一个 batch
            loss_rate = loss_counts.sum() / frame_samples[:, 1].sum() if loss_flags.any() else torch.tensor(0.0, device=device)
            
            # 构造模型输入（需要根据实际特征调整）
            frame_seq = frame_samples  # 假设 frame_samples 直接作为 frame_transformer 输入
            gcc_features = torch.cat([gcc_bitrate, rtt], dim=1)  # 组合 GCC 和 RTT 特征

            # 模型预测
            pred_bitrate, fec_ratios = model(
                frame_seq=frame_seq,
                loss_rate=loss_rate.unsqueeze(0).expand(batch_size, 1),  # 扩展维度匹配 batch
                rtt=rtt.unsqueeze(0).expand(batch_size, 1),
                gcc_features=gcc_features.unsqueeze(0).expand(batch_size, -1)
            )

            # ----------------------
            # 计算损失
            # ----------------------
            # 真实 GCC 比特率（从 batch 数据中获取）
            true_gcc_bitrate = gcc_bitrate.squeeze()

            # 计算损失
            loss = loss_fn(
                pred_bitrate=pred_bitrate,
                gcc_bitrate=true_gcc_bitrate,
                fec_table=fec_ratios,
                frame_samples=frame_samples,
                loss_flags=loss_flags,
                loss_counts=loss_counts
            )

            # ----------------------
            # 反向传播
            # ----------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失
            total_loss += loss.item()

            # 打印 batch 进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 打印 epoch 平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# 示例用法
if __name__ == "__main__":
    # 初始化组件
    fec_bins = torch.tensor([])  # 需与损失函数一致
    model = fecnet(frame_transformer=FrameTransformer())  # 替换为你的 frame_transformer
    loss_fn = OfflearningLoss(fec_bins)
    
    # 启动训练
    train(
        data_dir="",
        fec_bins=fec_bins,
        model=model,
        loss_fn=loss_fn,
        num_epochs=10,
        lr=1e-4,
        device="cuda"
    )