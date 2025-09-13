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

# 绘制图形
plt.plot(frame_size, loss_rate, label='loss', color='blue')
plt.plot(frame_size, fec_rate, label='fec', color='orange')
# 填充fec和loss之间的区域
plt.fill_between(frame_size, loss_rate, fec_rate, where=loss_rate <= fec_rate, interpolate=True, 
                 facecolor='green', alpha=0.3, hatch='//')

plt.xlabel('Frame Size')
plt.ylabel('Percent')
plt.title('Relationship between Frame Size, FEC Rate and Loss Rate')
plt.legend()
plt.show()