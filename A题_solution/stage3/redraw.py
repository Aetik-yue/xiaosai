"""
重绘图2(框架图)和图5(流程图)及修复红色方框问题
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False
})
OUT = r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\figures'
os.makedirs(OUT, exist_ok=True)

# ===== 图2: 框架图(重画,修复箭头对齐) =====
def draw_framework():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')

    # 三列节点 - 等宽等间距
    col_w, col_h = 3.2, 1.3
    cols_x = [2, 6, 10]  # 三列中心x
    top_y, bot_y = 5.0, 3.0
    
    colors = [
        ('#E6F1FB', '#185FA5'),  # 蓝
        ('#E1F5EE', '#0F6E56'),  # 绿
        ('#EEEDFE', '#534AB7'),  # 紫
    ]
    titles = ['运动学模型', '遮蔽判定模型', '优化模型']
    subtitles = [
        ['导弹 匀速直线', '无人机 水平等速', '烟幕弹 平抛', '云团 匀速下沉'],
        ['视锥 α=arcsin(R/d)', '完全遮蔽定理', '上下圆周判定', 'β_max ≤ α'],
        ['决策变量 θ,v,t,Δt', '目标 并集最大化', '约束 速度/时序/高度', 'PSO+DEGA求解'],
    ]

    for ci, (fc, ec) in enumerate(colors):
        x = cols_x[ci]
        # 上框 - 标题
        r1 = FancyBboxPatch((x-col_w/2, top_y-col_h/2), col_w, col_h,
                             boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.2)
        ax.add_patch(r1)
        ax.text(x, top_y, titles[ci], ha='center', va='center', fontsize=12, fontweight='bold', color=ec)

        # 下框 - 细节
        r2 = FancyBboxPatch((x-col_w/2, bot_y-col_h/2), col_w, col_h,
                             boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.0, alpha=0.7)
        ax.add_patch(r2)
        detail_text = '\n'.join(subtitles[ci])
        ax.text(x, bot_y, detail_text, ha='center', va='center', fontsize=10, color=ec)

    # 箭头: 列间连接(精确对齐)
    for y in [top_y, bot_y]:
        ax.annotate('', xy=(4.4, y), xytext=(3.6, y),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color='#888780'))
        ax.annotate('', xy=(8.4, y), xytext=(7.6, y),
                    arrowprops=dict(arrowstyle='->', lw=2.0, color='#888780'))

    # 标题和注释
    ax.text(6, 6.3, '三层递进建模体系', ha='center', fontsize=15, fontweight='bold', color='#2C2C2A')
    ax.text(6, 5.8, '五问共用运动学与遮蔽判定内核, 仅优化变量维度递进', ha='center', fontsize=10, color='#888780')
    ax.text(6, 1.8, 'Q1(验证)  →  Q2(单弹)  →  Q3(接力)  →  Q4(协同)  →  Q5(多对多)',
            ha='center', fontsize=11, fontweight='bold', color='#444441')

    plt.savefig(f'{OUT}/fig_framework.png'); plt.close()

# ===== 图5: PSO流程图(重画,修复箭头) =====
def draw_flowchart():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 8); ax.set_ylim(0, 9); ax.axis('off')

    col_w, col_h = 3.5, 1.0
    cx = 4

    steps = [
        (cx, 7.5, '初始化: 随机生成粒子群 (θ, v, t_rel, Δt)'),
        (cx, 6.0, '适应度评估: 离散时间步进仿真, 计算遮蔽时长 T'),
        (cx, 4.5, '更新个体/全局最优, PSO速度位置更新'),
        (cx, 3.0, '收敛判断: 是否达到迭代上限?'),
        (6.5, 0.8, '输出最优解'),
    ]

    for x, y, txt in steps:
        w = col_w if x == cx else 3.0
        h = col_h
        r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.12",
                           facecolor='#EEEDFE', edgecolor='#534AB7', lw=1.0)
        ax.add_patch(r)
        ax.text(x, y, txt, ha='center', va='center', fontsize=10, color='#3C3489')

    # 直线箭头(精确对齐)
    for y1, y2 in [(7.0, 6.5), (5.5, 5.0), (4.0, 3.5)]:
        ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='#888780'))

    # 分支线
    ax.plot([5.75, 6.5], [2.5, 2.5], 'k-', lw=1.2, color='#888780')
    ax.annotate('', xy=(6.5, 2.5), xytext=(5.75, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#888780'))
    
    # 从判断框右上角分叉
    ax.plot([5.75, 6.5], [3.0, 3.0], 'k-', lw=1.2, color='#888780')
    ax.annotate('', xy=(6.5, 3.0), xytext=(5.75, 3.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#888780'))

    # 标签
    ax.text(5.9, 2.85, '是', fontsize=11, fontweight='bold', color='#1D9E75')
    ax.text(5.9, 2.35, '否', fontsize=11, fontweight='bold', color='#D85A30')

    # 否的返回线(省略, 简化)
    # 从6.5向下到5.0再向左回到cx
    ax.plot([6.5, 6.5], [2.5, 5.0], 'r-', lw=1.2, alpha=0.5, color='#D85A30')
    ax.plot([6.5, 4], [5.0, 5.0], 'r-', lw=1.2, alpha=0.5, color='#D85A30')
    ax.annotate('', xy=(4, 5.0), xytext=(5.5, 5.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#D85A30'))
    ax.text(6.7, 3.8, '调整参数\n继续迭代', fontsize=9, color='#D85A30')

    # 是: 向下到输出
    ax.plot([cx, 6.5], [2.5, 0.8], 'g-', lw=1.5, alpha=0.5, color='#1D9E75')
    ax.annotate('', xy=(1.5, 0), xytext=(0, 0))  # dummy

    ax.text(4, 8.3, 'PSO优化算法流程', ha='center', fontsize=14, fontweight='bold', color='#2C2C2A')

    plt.savefig(f'{OUT}/fig_flowchart.png'); plt.close()

# ===== 修复红色方框: 重绘fig_summary(去掉bar的edge) =====
def draw_summary():
    qs = ['Q1\\n验证', 'Q2\\n单弹', 'Q3\\n3弹接力', 'Q4\\n3机协同', 'Q5\\n多对多']
    ours = [1.392, 4.578, 6.405, 11.506, 18.917]
    a2 = [1.3916, 4.587, 6.45, 11.549, 20.40]

    x = np.arange(5); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x-w/2, ours, w, label='本文', color='#378ADD', alpha=0.8, edgecolor='none')
    bars2 = ax.bar(x+w/2, a2, w, label='范文A2', color='#D85A30', alpha=0.6, edgecolor='none')

    for bar,v in zip(bars1,ours):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.3f}', ha='center', fontsize=9)
    for bar,v in zip(bars2,a2):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.2f}', ha='center', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(qs)
    ax.set_ylabel('有效遮蔽时长 (s)', fontsize=11)
    ax.set_title('Q1~Q5 结果与范文对比', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{OUT}/fig_summary.png'); plt.close()

if __name__ == '__main__':
    draw_framework(); print('fig_framework ✓')
    draw_flowchart(); print('fig_flowchart ✓')
    draw_summary(); print('fig_summary ✓')
