# 定义接收数据和发送数据的文件路径
import os
# script_dir = os.path.dirname(os.path.abspath(__file__))#试试这个当前脚本路径的吧。
script_path = os.path.abspath(__file__)
# 提取脚本所在目录
script_dir = os.path.dirname(script_path)

recv_data_path = os.path.join(script_dir, "recv_data.txt")
send_data_path = os.path.join(script_dir, "send_packet_data.txt")
#fout << packet.timestamp() << " " << seq << " " << packet.SequenceNumber(

# 定义SendData类，用于解析发送数据文件中的每一行
class SendData:
    def __init__(self, line: str):
        # 将行数据按空格分割
        splitted = line.split(" ")
        # 解析时间戳
        self.timestamp = int(splitted[0])
        # 解析序列号
        self.seq = int(splitted[1])
        # 解析流序列号
        self.stream_seq = int(splitted[2])
        # 解析负载类型
        self.payloadtype = splitted[3]
        pass
    pass

# 定义RecvData类，用于解析接收数据文件中的每一行
class RecvData:
    def __init__(self, line: str):
        # 将行数据按空格分割
        splitted = line.split(" ")
        # 解析序列号
        self.seq = int(splitted[0])
        # 解析发送时间
        self.sendtime = int(splitted[1])
        # 解析接收时间，如果为"lost"则设置为-1
        self.recvtime = -1 if splitted[2] == "lost" else int(splitted[2])
        # 解析最大反馈RTT
        self.max_feedback_rtt = int(splitted[3])
        # 解析最小传播RTT
        self.min_propagation_rtt = int(splitted[4])
        pass

    pass

# 初始化接收数据和发送数据的字典和列表
recv_data = {}
send_data = []

# 接收端反馈信息，在gcc下面
with open(recv_data_path, "r") as f:
    for line in f.readlines():
        # 去除行尾的换行符
        line = line.rstrip()
        # 解析每一行数据并创建RecvData对象
        packet = RecvData(line)
        # 将接收数据按序列号存入字典
        recv_data[packet.seq] = packet
        pass
    pass

#在egress下面，，时间戳一样说明是同一种包
with open(send_data_path, "r") as f:
    for line in f.readlines():
        # 去除行尾的换行符
        line = line.rstrip()
        # 解析每一行数据并创建SendData对象
        send_data.append(SendData(line))
        pass
    pass

# 打印接收数据和发送数据的数量
print(len(recv_data))
print(len(send_data))

# 定义Packet类，用于存储每个数据包的详细信息
class Packet:
    def __init__(self, seq: int, timestamp: int, payloadtype: str, lost: bool, recv_time: int, max_feedback_rtt, min_propagation_rtt):
        # 序列号
        self.seq = seq
        # 时间戳
        self.timestamp = timestamp
        # 负载类型
        self.payloadtype = payloadtype
        self.lost = lost
        self.recvtime = recv_time
        self.max_feedback_rtt = max_feedback_rtt
        self.min_propagation_rtt = min_propagation_rtt      
        pass

    pass

frames = []
new_frame = []

send_seq = 1
recv_seq = 1

# 遍历发送数据列表
for i in range(len(send_data)):
    send_packet = send_data[i]
    try:
        # 尝试从接收数据字典中获取对应的接收数据包
        recv_packet = recv_data[send_packet.seq]
        pass
    except Exception:
        # 如果获取失败，跳过当前循环
        continue
    # 如果负载类型为audio、padding、fec或retransmission，跳过当前循环
    if send_packet.payloadtype in [
        "audio",
        "padding",
        "fec",
        "retransmission",
    ]:
        continue
    # if send_packet.payloadtype == "fec":

    #     continue
    # 如果新帧列表不为空且当前数据包的时间戳与新帧列表中第一个数据包的时间戳不同
    #第二遍开始，send和recv到第二行，send和第一遍的新帧的时间戳不同，就会进入if了。
    #第一遍新帧加入，重置新帧，然后合成新帧。第三遍来就是把第二遍合的新帧加到总帧，以此重复，直到send结束
    if len(new_frame) != 0 and send_packet.timestamp != new_frame[0].timestamp:
        # 将新帧列表添加到帧列表中
        frames.append(new_frame)
        # 重置
        new_frame = []
        # 将当前数据包添加到新帧列表中
        new_frame.append(
            Packet(
                recv_packet.seq,
                send_packet.timestamp,
                send_packet.payloadtype,
                recv_packet.recvtime == -1,
                recv_packet.recvtime,
                recv_packet.max_feedback_rtt,
                recv_packet.min_propagation_rtt
            )
        )

    else:
        # 首次，会走这里，send在第一个，合一个新帧，
        new_frame.append(
            Packet(
                recv_packet.seq,
                send_packet.timestamp,
                send_packet.payloadtype,
                recv_packet.recvtime == -1,
                recv_packet.recvtime,
                recv_packet.max_feedback_rtt,
                recv_packet.min_propagation_rtt
            )
        )
        pass
    pass

# 打印帧的数量
print(len(frames))

# 定义函数loss_num，用于计算帧中丢失的数据包数量
def loss_num(frame):
    ret = 0
    for f in frame:
        if f.lost:
            ret += 1
            pass
        pass
    return ret
    
# 定义函数计算帧的平均max_feedback_rtt
def avg_rtt(frame):
    total_rtt=0
    for f in frame:
        total_rtt += f.max_feedback_rtt
        rtt=total_rtt // len(frame)
        pass
    return rtt 


# 定义输出文件路径
output_path = os.path.join(script_dir,"frame_loss_rtt.txt")

# 打开输出文件
out = open(output_path, "w")

# 遍历帧列表，将每帧的数据包数量,fec数,和丢失数量写入输出文件
for i in frames:
    out.write(str(len(i))+' '+str(loss_num(i))+' '+str(avg_rtt(i)) )
    out.write('\n')