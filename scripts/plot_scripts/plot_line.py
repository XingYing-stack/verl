import matplotlib.pyplot as plt
import numpy as np

# 数据
sampling_ratio = [0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0]
correct_rate = [0.160, 0.163, 0.185, 0.242, 0.360, 0.408, 0.360]
efficiency = [0.800, 0.936, 0.877, 0.937, 0.913, 0.948, 0.991]

# 设置字体与样式（论文级别）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'pdf.fonttype': 42,   # 确保矢量字体嵌入
    'ps.fonttype': 42,
})

# 绘图
fig, ax1 = plt.subplots(figsize=(6, 4))

# 颜色方案（简洁高对比）
color1 = '#1f77b4'  # 蓝色
color2 = '#d62728'  # 红色

# 曲线1: Correct Rate
ln1 = ax1.plot(sampling_ratio, correct_rate, marker='o', color=color1, linewidth=2.2, label='Correct Rate')
ax1.set_xlabel('Sampling Ratio')
ax1.set_ylabel('Correct Rate', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 曲线2: Efficiency（双轴）
ax2 = ax1.twinx()
ln2 = ax2.plot(sampling_ratio, efficiency, marker='s', linestyle='--', color=color2, linewidth=2.2, label='Efficiency')
ax2.set_ylabel('Efficiency', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 合并图例
lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='lower right', frameon=False)

# 样式细节优化
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(0.1, 0.45)
ax2.set_ylim(0.75, 1.02)
ax1.set_title('Effect of Sampling Ratio on RL Performance', pad=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# 布局优化
plt.tight_layout()
plt.savefig("sampling_ratio_performance.pdf", bbox_inches='tight')  # 保存矢量图用于论文
plt.show()
