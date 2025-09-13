1，源数据
recv_date.txt包含序列号，发送时间，包接收时间，max反馈rtt，min传播rtt
send_packet_data.txt：
时间戳，seq，包序列号，输出包类型（audio，video，重传，padding，fec，unknow），
包大小，key还是delta帧是不是帧的首包，或者是fec时打印是否帧首包（begin）
2，处理脚本
frame_fec_loss：处理send和recv数据得到——帧大小、fec包数、loss数。用于之后绘制观察当前fec和loss是否合适
frame_loss_rtt：得到帧大小、loss数、rtt。用作训练。
3，画图的东西
green-pic.py：用于绘制以帧大小为横坐标，fec和loss的情况，其中，如果出现丢包才会算作有效数据，
同样的帧大小下会计算丢包平均数和fec。
（好像不对，应该绘制fec和loss的曲线，横坐标从1开始递增，不然出现fec不够时，因为算的平均，fec数也会够）



