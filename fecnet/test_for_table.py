from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def evaluate_table_fec(input_file, output_file=None,four_column_format=True):
    """
    评估表查询方式的FEC覆盖率和过保护程度
    
    Args:
        input_file: 输入文件路径，包含帧大小、fec和loss三列
        output_file: 可选，输出评估报告的文件路径
    """
    print(f"读取数据文件: {input_file}")
    rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 读取输入数据 (帧大小, fec, loss)
    if four_column_format:
        # 四列格式：帧大小、loss、rtt、fec
        df = pd.read_csv(input_file, sep='\s+', header=None, names=["frame_size", "loss", "rtt", "fec"])
        print("使用四列数据格式: 帧大小, loss, rtt, fec")
    else:
        # 三列格式：帧大小、fec、loss
        df = pd.read_csv(input_file, sep='\s+', header=None, names=["frame_size", "fec", "loss"])
        print("使用三列数据格式: 帧大小, fec, loss")
    
    print(f"数据总行数: {len(df)}")
    
    # 基本统计
    print("\n=== 基本统计 ===")
    print(f"平均帧大小: {df['frame_size'].mean():.2f}")
    print(f"平均FEC包数: {df['fec'].mean():.2f}")
    print(f"平均丢包数: {df['loss'].mean():.2f}")
    
    # 计算有效数据 (有丢包的帧)
    loss_frames = df[df["loss"] > 0].copy()
    print(f"有丢包的帧数: {len(loss_frames)} ({len(loss_frames)/len(df)*100:.2f}%)")
    
    if len(loss_frames) > 0:
        print("\n=== FEC性能评估 ===")
        
        # 计算FEC覆盖率 (FEC >= loss)
        coverage = (loss_frames["fec"] >= loss_frames["loss"]).mean() * 100
        print(f"FEC覆盖率: {coverage:.2f}% (FEC >= 实际丢包)")
        
        # 计算过保护比率 (FEC / loss)
        loss_frames["protection_ratio"] = loss_frames["fec"] / loss_frames["loss"].clip(lower=1)
        over_protection_ratio = loss_frames["protection_ratio"].mean()
        print(f"平均过保护比率: {over_protection_ratio:.2f}倍 (FEC/实际丢包)")
        
        # 计算过保护绝对值 (FEC - loss)
        loss_frames["protection_absolute"] = loss_frames["fec"] - loss_frames["loss"]
        over_protection_absolute = loss_frames["protection_absolute"].mean()
        print(f"平均过保护绝对值: {over_protection_absolute:.2f}包 (FEC-实际丢包)")
        
        # 计算过保护和欠保护的分布
        over_protected = loss_frames[loss_frames["fec"] > loss_frames["loss"]]
        under_protected = loss_frames[loss_frames["fec"] < loss_frames["loss"]]
        perfect_protected = loss_frames[loss_frames["fec"] == loss_frames["loss"]]
        
        print(f"过度保护的帧数: {len(over_protected)} ({len(over_protected)/len(loss_frames)*100:.2f}%)")
        print(f"保护不足的帧数: {len(under_protected)} ({len(under_protected)/len(loss_frames)*100:.2f}%)")
        print(f"完美保护的帧数: {len(perfect_protected)} ({len(perfect_protected)/len(loss_frames)*100:.2f}%)")
        
        # 按帧大小区间分析
        print("\n=== 按帧大小区间分析 ===")
        # 创建大小区间
        bins = [0, 10, 20, 50, 100, 200, np.inf]
        labels = ['0-10', '11-20', '21-50', '51-100', '101-200', '>200']
        loss_frames['size_bin'] = pd.cut(loss_frames['frame_size'], bins=bins, labels=labels)
        
        size_analysis = loss_frames.groupby('size_bin').agg({
            'frame_size': 'count',
            'protection_ratio': 'mean',
            'protection_absolute': 'mean',
            'loss': 'mean',
            'fec': 'mean'
        }).reset_index()
        
        size_analysis['coverage'] = loss_frames.groupby('size_bin').apply(
            lambda x: (x['fec'] >= x['loss']).mean() * 100
        ).values
        
        size_analysis.rename(columns={
            'frame_size': '帧数量',
            'protection_ratio': '平均过保护比率',
            'protection_absolute': '平均过保护绝对值',
            'loss': '平均丢包数',
            'fec': '平均FEC数',
            'coverage': 'FEC覆盖率(%)'
        }, inplace=True)
        
        # 打印区间分析结果
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(size_analysis.to_string(index=False))
        
        # 可视化结果
        create_visualizations(loss_frames, output_file)
        
        # 保存详细结果
        if output_file:
            # 保存详细数据
            details_file = os.path.splitext(output_file)[0] + "_details.csv"
            loss_frames.to_csv(details_file, index=False)
            print(f"\n详细分析结果已保存至: {details_file}")
            
            # 保存区间分析
            bins_file = os.path.splitext(output_file)[0] + "_bins.csv"
            size_analysis.to_csv(bins_file, index=False)
            print(f"区间分析结果已保存至: {bins_file}")
    else:
        print("\n数据集中没有丢包帧，无法评估FEC性能")

def create_visualizations(loss_frames, output_file=None):
    """创建可视化图表"""
    plt.figure(figsize=(15, 10))
    
    # 1. FEC vs Loss 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(loss_frames["loss"], loss_frames["fec"], alpha=0.5)
    plt.plot([0, loss_frames["loss"].max()], [0, loss_frames["loss"].max()], 'r--')  # 对角线
    plt.xlabel("实际丢包数")
    plt.ylabel("FEC包数")
    plt.title("FEC vs 实际丢包")
    plt.grid(True, alpha=0.3)
    
    # 2. 过保护比率随帧大小变化
    plt.subplot(2, 2, 2)
    plt.scatter(loss_frames["frame_size"], loss_frames["protection_ratio"].clip(upper=5), alpha=0.5)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel("帧大小")
    plt.ylabel("过保护比率 (FEC/Loss)")
    plt.title("过保护比率随帧大小变化 (上限5倍)")
    plt.grid(True, alpha=0.3)
    
    # 3. 过保护绝对值直方图
    plt.subplot(2, 2, 3)
    plt.hist(loss_frames["protection_absolute"], bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("过保护绝对值 (FEC-Loss)")
    plt.ylabel("频率")
    plt.title("过保护绝对值分布")
    plt.grid(True, alpha=0.3)
    
    # 4. 按帧大小区间的覆盖率
    bins = [0, 10, 20, 50, 100, 200, np.inf]
    labels = ['0-10', '11-20', '21-50', '51-100', '101-200', '>200']
    loss_frames['size_bin'] = pd.cut(loss_frames['frame_size'], bins=bins, labels=labels)
    
    coverage_by_bin = loss_frames.groupby('size_bin').apply(
        lambda x: (x['fec'] >= x['loss']).mean() * 100
    )
    
    plt.subplot(2, 2, 4)
    coverage_by_bin.plot(kind='bar')
    plt.xlabel("帧大小区间")
    plt.ylabel("FEC覆盖率(%)")
    plt.title("不同帧大小区间的FEC覆盖率")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file)
        print(f"可视化结果已保存至: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估表查询方式的FEC覆盖率和过保护程度")
    parser.add_argument("input_file", help="输入文件路径，包含帧大小、fec和loss三列")
    parser.add_argument("-o", "--output", help="输出评估报告和图表的文件路径")
    parser.add_argument("-f", "--format", choices=["3col", "4col"], default="4col",
                       help="数据格式: 3col(帧大小,fec,loss) 或 4col(帧大小,loss,rtt,fec), 默认为4col")
    args = parser.parse_args()
    
    evaluate_table_fec(args.input_file, args.output, four_column_format=(args.format == "4col"))