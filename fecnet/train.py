# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from config import config  # 导入配置
from FrameTransformer import FrameTransformer
from Fecnet import fecnet
from Dataset import OfflearningDataset, collate_fn
import pdb
from Multiloss import OfflearningLoss

def train():
    dataset = OfflearningDataset(config.data_dir, frames_per_group=config.frames_per_group)
    pdb.set_trace()
    frame_transformer = FrameTransformer(
        d_model=config.frame_transformer_params["d_model"],
        nhead=config.frame_transformer_params["nhead"],
        num_layers=config.frame_transformer_params["num_layers"],
        dim_feedforward=config.frame_transformer_params["dim_feedforward"]
    )
    
    # 创建FecNet模型
    model = fecnet(
        frame_transformer,
        input_dim=config.fecnet_params["input_dim"]  # 使用input_dim替代gcc_input_dim
    ).to(config.device)
    
    loss_fn = OfflearningLoss(config.fec_bins).to(config.device)
    
    optimizer = Adam(
        model.parameters(),
        lr=config.optimizer_params["lr"],
        betas=config.optimizer_params["betas"],
        weight_decay=config.optimizer_params["weight_decay"]
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=config.scheduler_params["step_size"],
        gamma=config.scheduler_params["gamma"]
    )

    train_loop(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        checkpoint_dir=config.checkpoint_dir,
        resume_checkpoint=config.resume_checkpoint
    )

def train_loop(
    model, dataset, loss_fn, optimizer, scheduler, device,
    num_epochs, batch_size, checkpoint_dir, resume_checkpoint
):
    os.makedirs(checkpoint_dir, exist_ok=True)
     # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,  # 使用多进程加载数据
        pin_memory=True  # 加速数据传输到GPU
    )
    
    # 如果有检查点，从检查点恢复
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # 将数据移到指定设备
            frame_samples = batch["frame_samples"].to(device)
            loss_flags = batch["loss_flags"].to(device)
            loss_counts = batch["loss_counts"].to(device)
            rtt = batch["rtt"].to(device)
            avg_loss_rate = batch["avg_loss_rate"].to(device)
            mask = batch["mask"].to(device)

            pred_bitrate, fec_table = model(
                frame_samples, 
                avg_loss_rate,  # 使用平均丢包率
                rtt
            )

            # 计算损失
            loss = loss_fn(
                pred_bitrate, 
                fec_table, 
                frame_samples, 
                loss_flags, 
                loss_counts, 
                rtt.squeeze(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 加一个保存最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss
            }, best_checkpoint_path)
            print(f"Best model saved to {best_checkpoint_path}")



        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()