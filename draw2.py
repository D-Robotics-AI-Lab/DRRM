import numpy as np

data = {
    'DP': np.array([0.7,77.7,39.3,14,69.3,31,63,73.7,63.3,7.3,19.3,4.7,20.0,3.7]),
    'DP3': np.array([79.3, 97.7, 85.3, 83.7, 88.7, 18.7, 84.7, 83.3, 64.0, 60.7, 56.3, 13.7, 58.3, 22.0]),
    'VO-DP': np.array([85.0, 89.7, 63.3, 43.0, 82.0, 80.0, 94.3, 88.3, 67.3, 32.3, 43.0, 17.0, 58.3, 52.3]),
    'VO-DP-1': np.array([78.7, 94.7, 69.3, 31.3, 77.3, 81.7, 98.0, 86.3, 60.3, 31.3, 52.0, 19.3, 55.3, 69.3])
}
ms = []
for k, d in data.items():
    print(f"{k}: mean {d.mean()}, std { d.std()}")
    ms.append((d.mean(), d.std()))
means, stds = zip(*ms)

import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 900,
    'savefig.dpi': 900,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 数据
methods = ['DP', 'DP3', 'VO-DP', 'VO-DP-1']
# means = [34.785714285714285, 64.02857142857142, 63.98571428571427, 64.6285714285714]
# stds = [27.864797483123937, 26.821458055909225, 23.09408882756603, 23.486696190805205]

# 颜色设置
method_styles = {
    "DP": {"color": "#c6dbef"},
    "DP3": {"color": "#fc9271"},
    "VO-DP": {"color": "#6baed6"},
    "VO-DP-1": {"color": "#6baed6"}
}

# 创建图表
fig, ax = plt.subplots(figsize=(10, 7))

# 设置柱状图位置
x_pos = np.arange(len(methods))
bar_width = 0.6

# 绘制柱状图
colors = [method_styles[method]["color"] for method in methods]
bars = ax.bar(x_pos, means, yerr=stds, capsize=5, width=bar_width, 
              color=colors, edgecolor='black', linewidth=0, error_kw={'elinewidth': 2})

# 设置图表属性
ax.set_xlabel('Methods', fontsize=25)
ax.set_ylabel('Success Rate (%)', fontsize=25)
# ax.set_title('Performance Comparison with Error Bars', fontweight='bold', pad=15, fontsize=30)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=30, )
ax.grid(True, linestyle='--', alpha=0.7, axis='y')
# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

# 在VO-DP-1顶部添加五角星标记
ax.plot(3, means[3], marker='*', markersize=40, color='#6baed6', markeredgecolor='white', markeredgewidth=2)

# 在柱状图上添加数值标签
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + 3, f'{mean:.1f}', ha='center', va='bottom', fontsize=30, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0', facecolor='white', alpha=0.9, edgecolor='none'))

# 用虚线连接柱子顶部
ax.plot(x_pos, means, linestyle='--', color='gray', linewidth=2, marker='None')

# 保存为PDF
filename = 'out/figures/simulator.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
filename = 'out/figures/simulator.png'
plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.1)
filename = 'out/figures/simulator.svg'
plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0.1)
print(filename)

plt.show()