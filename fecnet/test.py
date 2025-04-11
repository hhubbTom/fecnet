# import torch
# from config import config
# from Dataset import OfflearningDataset, collate_fn
# from FrameTransformer import FrameTransformer
# from Fecnet import fecnet
# from torch.utils.data import DataLoader

# def test_pipeline():
#     """测试整个数据处理和模型推理流程"""
#     # 加载数据集
#     print("Loading dataset...")
#     dataset = OfflearningDataset(config.data_dir, frames_per_group=config.frames_per_group)
#     print(f"Dataset size: {len(dataset)}")
    
#     # 测试单个样本
#     print("\nTesting single sample...")
#     sample = dataset[0]
#     for key, value in sample.items():
#         if isinstance(value, torch.Tensor):
#             print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#         else:
#             print(f"{key}: {value}")
    
#     # 测试数据加载器
#     print("\nTesting dataloader...")
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#     batch = next(iter(dataloader))
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#         else:
#             print(f"{key}: {value}")
    
#     # 测试模型
#     print("\nTesting model...")
#     device = torch.device("cpu")  # 使用CPU进行测试
    
#     # 创建模型
#     frame_transformer = FrameTransformer(
#         d_model=config.frame_transformer_params["d_model"],
#         nhead=config.frame_transformer_params["nhead"],
#         num_layers=config.frame_transformer_params["num_layers"],
#         dim_feedforward=config.frame_transformer_params["dim_feedforward"]
#     )
    
#     model = fecnet(
#         frame_transformer,
#         input_dim=config.fecnet_params["input_dim"]
#     ).to(device)
    
#     # 准备输入数据
#     frame_samples = batch["frame_samples"].to(device)
#     avg_loss_rate = batch["avg_loss_rate"].to(device)
#     rtt = batch["rtt"].to(device)
    
#     # 模型推理
#     with torch.no_grad():
#         bitrate, fec_table = model(frame_samples, avg_loss_rate, rtt)
    
#     # 打印输出
#     print(f"Predicted bitrate: shape={bitrate.shape}, values={bitrate}")
#     print(f"Predicted FEC table: shape={fec_table.shape}")
#     print(f"FEC table sample:\n{fec_table[0]}")

# if __name__ == "__main__":
#     test_pipeline()