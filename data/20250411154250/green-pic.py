import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
# 替换为系统中存在的中文字体路径，例如 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 更改工作目录
os.chdir(script_dir)
# 读取txt文件数据，假设数据以空格分隔
data = np.loadtxt('frame_fec_loss.txt')
frame_size = data[:, 0]  # 第一列帧大小
fec_rate = data[:, 1]  # 第二列fec率
loss_rate = data[:, 2]  # 第三列丢包率

# 筛选有效数据（loss_rate > 0）
valid_indices = (loss_rate > 0) & (frame_size > 1)
valid_frame_size = frame_size[valid_indices]
valid_fec_rate = fec_rate[valid_indices]
valid_loss_rate = loss_rate[valid_indices]

# 按帧大小分组，计算平均FEC率和平均丢包率
unique_frame_sizes = np.unique(valid_frame_size)
average_fec_percentage = [
    (valid_fec_rate[valid_frame_size == size].mean() / size) * 100 for size in unique_frame_sizes
]
average_loss_percentage = [
    (valid_loss_rate[valid_frame_size == size].mean() / size) * 100 for size in unique_frame_sizes
]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(unique_frame_sizes, average_loss_percentage, linestyle='-', color='#1E90FF', label='loss')
plt.plot(unique_frame_sizes, average_fec_percentage, linestyle='-', color='orange', label='fec')

# # 填充fec和loss之间的区域
# plt.fill_between(unique_frame_sizes, average_loss_percentage, average_fec_percentage, 
#                 where=np.array(average_loss_percentage) <= np.array(average_fec_percentage), 
#                 interpolate=True, facecolor='green', alpha=0.3, hatch='//')

# 设置横坐标从负数开始
plt.xlim(left=-5)  # 横坐标从 -5 开始
plt.ylim(bottom=0, top=80)  # 纵坐标范围设置为 0 到 80

# 设置纵坐标刻度为 10% 的间隔
plt.yticks(np.arange(0, 81, 10))  # 从 0 到 80，每隔 10 设置一个刻度
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))  # 精确到小数点后一位

plt.xlabel('Frame Size')
plt.ylabel('Percent')
plt.title('不同帧大小下的FEC率和丢包率')
plt.legend()

plt.tight_layout()
plt.show()