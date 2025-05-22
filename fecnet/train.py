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
from Dataset import OfflearningDataset
import pdb
import matplotlib.pyplot as plt  # 添加导入

from Multiloss import OfflearningLoss

def train():
    dataset = OfflearningDataset(config.data_dir)
    # 在这里添加断点，查看第一个样本
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        print(f"{key}:", value)
    #pdb.set_trace()  # 添加断点

    frame_transformer = FrameTransformer(
        d_model=config.frame_transformer_params["d_model"],
        nhead=config.frame_transformer_params["nhead"],
        num_layers=config.frame_transformer_params["num_layers"],
        dim_feedforward=config.frame_transformer_params["dim_feedforward"]
    )
    #pdb.set_trace()  # 添加断点，查看FrameTransformer的结构
    # 创建FecNet模型
    model = fecnet(
        frame_transformer,
        #input_dim=config.fecnet_params["input_dim"]  # input_dim为2
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
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,  
        pin_memory=True
    )
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    best_loss = float('inf') 
    loss_history = []  # 用于记录每个 epoch 的平均损失

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            frame_samples = batch["frames"].to(device)
            loss_frames = batch["loss_frames"].to(device)
            loss = batch["loss"].to(device)
            rtt = batch["rtt"].to(device)
            avg_loss = batch["avg_loss"].to(device)

            fec_table = model(
                frame_samples, 
                avg_loss,  
                rtt
            )
            
            Loss = loss_fn(
                fec_table,  
                loss_frames, 
                loss.squeeze(-1) 
            )

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            total_loss += Loss.item()
            progress_bar.set_postfix(Loss=Loss.item())

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # 记录平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

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

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))  # 保存损失曲线
    plt.show()

if __name__ == "__main__":
    train()