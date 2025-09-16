import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式，优化用于LaTeX文档
plt.rcParams.update({
    'font.size': 16,           # 增大字体
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 16,      # 增大坐标轴标签字体
    'axes.titlesize': 18,      # 增大标题字体
    'xtick.labelsize': 14,     # 增大x轴刻度字体
    'ytick.labelsize': 14,     # 增大y轴刻度字体
    'legend.fontsize': 14,     # 增大图例字体
    'lines.linewidth': 3,      # 加粗线条
    'lines.markersize': 12,    # 加大标记
    'figure.dpi': 900,         # 高分辨率
    'savefig.dpi': 900,        # 保存高分辨率图像
    'savefig.bbox': 'tight',   # 紧凑边界
    'savefig.pad_inches': 0.1  # 减少空白边缘
})

# 演示数量（添加0作为起点）
demos = [0, 20, 50, 100]

# 每个任务的数据 (DP, DP3, VO-DP)，添加0演示时的成功率为0
tasks = {
    "Pick Apple Messy": {
        "DP": [0, 5.3, 16.7, 31.0],
        "DP3": [0, 4.0, 12.7, 18.7],
        "VO-DP": [0, 3.0, 5.0, 80.0]
    },
    "Block Hammer Beat": {
        "DP": [0, 0.0, 0.0, 0.7],
        "DP3": [0, 55.7, 64.7, 79.3],
        "VO-DP": [0, 4.7, 37.0, 85.0]
    },
    "Dual Bottles Pick (Easy)": {
        "DP": [0, 1.7, 38.3, 73.7],
        "DP3": [0, 40.3, 74.7, 83.3],
        "VO-DP": [0, 10.7, 61.3, 88.3]
    },
    "Put Apple Cabinet": {
        "DP": [0, 2.3, 38.3, 63.6],
        "DP3": [0, 50.0, 73.3, 84.7],
        "VO-DP": [0, 61.7, 77.0, 94.3]
    },
    "AVG.": {
        "DP": [0, 2.3, 23.3, 42.3],
        "DP3": [0, 37.5, 56.4, 66.5],
        "VO-DP": [0, 20.0, 45.1, 86.9]
    }
}

# 为不同方法设置颜色和标记样式
method_styles = {
    "DP": {"color": "#c6dbef", "marker": "X", "linestyle": "-", "linewidth": 7, "markersize": 25},
    "DP3": {"color": "#fc9271", "marker": "X", "linestyle": "-", "linewidth": 7, "markersize": 25},
    "VO-DP": {"color": "#6baed6", "marker": "X", "linestyle": "-", "linewidth": 7, "markersize": 25}
}

# 为每个任务创建单独的图表并保存为PDF
filenames = []
for task_name, task_data in tasks.items():
    # 创建新图表，增大图形尺寸以便在并排时更清晰
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制每条线
    lines = []
    labels = []
    for method, values in task_data.items():
        style = method_styles[method]
        line, = ax.plot(demos, values, 
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markeredgewidth=2,        # 设置标记边框宽度
                markeredgecolor='white')  # 设置标记边框颜色为白色
        lines.append(line)
        labels.append(method)
    
    # 设置图表属性，优化用于LaTeX文档
    # ax.set_title(task_name, fontweight='bold', pad=15, fontsize=30)
    ax.set_xlabel('Number of Demonstrations', fontsize=25)
    if task_name=='Pick Apple Messy':
        ax.set_ylabel('Success Rate (%)', fontsize=25)
    # ax.set_xticks(demos, fontsize=30)
    # ax.set_yticks(fontsize=30)
    ax.set_xticks(demos)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_ylim(-5, 105)
    ax.set_xlim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    # 优化图例位置和样式
    if task_name=='Pick Apple Messy' or 'AVG.' == task_name:
        legend_lines = [plt.Line2D([0], [0], color=method_styles[method]["color"], 
                              linewidth=method_styles[method]["linewidth"], 
                              linestyle=method_styles[method]["linestyle"]) 
                   for method in labels]
        ax.legend(
            legend_lines, labels,
            loc='upper left', 
            frameon=True, 
            fancybox=True, 
            shadow=False, 
            # numpoints=0,
            fontsize=30,  # 增加图例字体大小
            framealpha=0.5,  # 设置图例背景透明度
            handletextpad=0.5,  # 调整图例文本与标记间距
            columnspacing=1,  # 调整图例列间距
            handlelength=1.5  # 调整图例标记长度
        )
    
    # 保存为PDF，优化用于LaTeX文档
    # filename = f'out/figures/{"".join([n[0].upper() for n in task_name.replace(".", "").replace("(", "").replace(")", "").split(" ")])}.pdf'
    # plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
    filename = f'out/figures/{"".join([n[0].upper() for n in task_name.replace(".", "").replace("(", "").replace(")", "").split(" ")])}.png'
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.1)
    # filename = f'out/figures/{"".join([n[0].upper() for n in task_name.replace(".", "").replace("(", "").replace(")", "").split(" ")])}.svg'
    # plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0.1)
    filenames.append(filename)
    plt.close()  # 关闭当前图表以释放内存

# 打印生成的文件名
for filename in filenames:
    print(filename)