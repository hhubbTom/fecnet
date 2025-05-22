import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from FrameTransformer import FrameTransformer
from Fecnet import fecnet
from config import config

def predict_fec(model_path, input_file, output_file):
    """
    对输入文件进行FEC预测，并将结果保存到输出文件
    
    Args:
        model_path: 训练好的模型路径
        input_file: 输入文件路径，包含帧大小、loss和rtt三列
        output_file: 输出文件路径，将添加第四列FEC预测值
    """
    print(f"加载模型: {model_path}")
    
    # 创建模型架构
    frame_transformer = FrameTransformer(
        d_model=config.frame_transformer_params["d_model"],
        nhead=config.frame_transformer_params["nhead"],
        num_layers=config.frame_transformer_params["num_layers"],
        dim_feedforward=config.frame_transformer_params["dim_feedforward"]
    )
    
    model = fecnet(frame_transformer).to(config.device)
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"加载输入数据: {input_file}")
    # 读取输入数据
    df = pd.read_csv(input_file, sep='\s+', header=None, names=["frame_size", "loss", "rtt"])
    
    # 创建结果DataFrame
    result_df = df.copy()
    result_df["predicted_fec"] = 0.0  # 添加第四列用于存储预测的FEC值
    
    print("开始预测...")
    chunk_size = 10  # 与训练时相同的块大小
    
    # 确保数据行数是chunk_size的整数倍
    num_full_chunks = len(df) // chunk_size
    
    with torch.no_grad():  # 禁用梯度计算以加快预测速度
        # 处理完整的chunk
        for i in tqdm(range(num_full_chunks), desc="预测进度"):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = df.iloc[start_idx:end_idx]
            
            # 提取这个chunk的数据
            frames = torch.tensor(chunk["frame_size"].values, dtype=torch.float32).unsqueeze(1)
            frames = frames.unsqueeze(0)  # 添加批次维度 [1, 10, 1]
            
            mean_loss = chunk["loss"].mean()
            avg_loss = torch.tensor([[mean_loss]], dtype=torch.float32)
            
            mean_rtt = chunk["rtt"].mean()
            rtt = torch.tensor([[mean_rtt]], dtype=torch.float32)
            
            # 移至设备
            frames = frames.to(config.device)
            avg_loss = avg_loss.to(config.device)
            rtt = rtt.to(config.device)
            
            # 获取预测的FEC表
            fec_table = model(frames, avg_loss, rtt)  # [1, 10]
            
            # 对每个帧应用预测的FEC值
            for j in range(chunk_size):
                frame_idx = start_idx + j
                frame_size = df.iloc[frame_idx]["frame_size"]
                
                # 将帧大小映射到FEC表的索引
                bin_idx = torch.bucketize(torch.tensor([frame_size]), config.fec_bins, right=True)
                bin_idx = torch.clamp(bin_idx, max=len(config.fec_bins)-1).item()
                
                # 获取预测的FEC比率
                fec_rate = fec_table[0, bin_idx].item()
                
                # 计算FEC包数量 (FEC比率 * 帧大小)
                predicted_fec_count = round(fec_rate * frame_size)
                
                # 保存预测结果
                result_df.at[frame_idx, "predicted_fec"] = predicted_fec_count
    
    # 处理剩余的不足一个chunk的数据
    remaining_rows = len(df) % chunk_size
    if remaining_rows > 0:
        print(f"注意: 有{remaining_rows}行数据不足一个完整chunk，将使用默认FEC率。")
        # 可以采用一个默认的FEC率，或者使用训练数据的平均FEC率
        # 这里简单地使用0.1作为默认FEC率
        default_fec_rate = 0.1
        start_idx = num_full_chunks * chunk_size
        for i in range(remaining_rows):
            frame_idx = start_idx + i
            frame_size = df.iloc[frame_idx]["frame_size"]
            result_df.at[frame_idx, "predicted_fec"] = default_fec_rate * frame_size
    
    # 保存结果
    result_df.to_csv(output_file, sep=' ', header=False, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    # 评估结果
    print("\n=== 预测评估 ===")
    
    # 计算FEC覆盖率
    loss_frames = df[df["loss"] > 0]
    if len(loss_frames) > 0:
        # 对应索引的预测FEC
        predicted_fec_for_loss = result_df.loc[loss_frames.index, "predicted_fec"]
        
        # 计算覆盖率
        coverage = (predicted_fec_for_loss >= loss_frames["loss"]).mean() * 100
        print(f"FEC覆盖率: {coverage:.2f}% (预测FEC >= 实际丢包)")
        
        # 计算过保护程度
        over_protection_ratio = (predicted_fec_for_loss / loss_frames["loss"].clip(lower=1)).mean()
        print(f"平均过保护比率: {over_protection_ratio:.2f}倍 (预测FEC/实际丢包)")
        
        over_protection_absolute = (predicted_fec_for_loss - loss_frames["loss"]).mean()
        print(f"平均过保护绝对值: {over_protection_absolute:.2f}包 (预测FEC-实际丢包)")
    else:
        print("测试数据中没有丢包帧，无法计算覆盖率和过保护程度")

if __name__ == "__main__":
    # 配置路径
    model_path = "checkpoints/best_model.pt"  # 您的最佳模型路径
    input_file = "data/test_data.txt"  # 您的测试数据文件路径，三列：帧大小、loss、rtt
    output_file = "data/test_data_with_fec.txt"  # 输出文件路径，四列：帧大小、loss、rtt、预测FEC
    
    predict_fec(model_path, input_file, output_file)