import numpy as np
data = {
    'DP': np.array([23.3, 16.7, 3.3, 1.7]),
    'DP3': np.array([73.3, 68.3, 75.0, 53.3]),
    'VO-DP-1': np.array([96.7, 91.6, 93.3, 70.0])
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
methods = ['DP', 'DP3', 'VO-DP-1']
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
# ax.set_ylabel('Success Rate (%)', fontsize=25)
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
ax.plot(2, means[2], marker='*', markersize=40, color='#6baed6', markeredgecolor='white', markeredgewidth=2)

# 在柱状图上添加数值标签
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + 1, f'{mean:.1f}', ha='center', va='bottom', fontsize=30, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0', facecolor='white', alpha=0.5, edgecolor='none'))

# 用虚线连接柱子顶部
ax.plot(x_pos, means, linestyle='--', color='gray', linewidth=2, marker='None')



# 保存为PDF
filename = 'out/figures/realworld.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
filename = 'out/figures/realworld.png'
plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.1)
filename = 'out/figures/realworld.svg'
plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0.1)
print(filename)

plt.show()