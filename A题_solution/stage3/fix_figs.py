"""修复fig_framework和fig_flowchart"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys

sys.path.insert(0, r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False
})
OUT = r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\figures'

# ===== 修复 fig_framework: 三层模型框架 =====
def fig_framework_v2():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis('off')

    # 三列布局
    boxes = [
        ('运动学模型\n(第4.1节)', 1, 4.5, '#E6F1FB', '#185FA5'),
        ('导弹 匀速直线\n无人机 水平等速\n烟幕弹 平抛\n云团 匀速下沉', 1, 2.8, '#E6F1FB', '#185FA5'),
        ('遮蔽判定模型\n(第4.2节 · 核心)', 4.5, 4.5, '#E1F5EE', '#0F6E56'),
        ('视锥定义 α=arcsin(R/d)\n完全遮蔽定理\n上下圆周判定\nβ_max≤α', 4.5, 2.8, '#E1F5EE', '#0F6E56'),
        ('优化模型\n(第4.3节)', 8, 4.5, '#EEEDFE', '#534AB7'),
        ('决策变量: θ,v,t_rel,Δt\n目标: 并集最大化\nPSO+DEGA求解', 8, 2.8, '#EEEDFE', '#534AB7'),
    ]
    for txt, x, y, fc, ec in boxes:
        w, h = 3.2, 1.4
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                               facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, txt, ha='center', va='center', fontsize=10, color=ec)

    # 箭头
    for y in [4.5, 3.2]:
        ax.annotate('', xy=(4.5, y), xytext=(2.6, y),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='gray'))
        ax.annotate('', xy=(8, y), xytext=(6.1, y),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='gray'))

    # 标题
    ax.text(5.5, 6.1, '三层递进建模体系', ha='center', fontsize=14, fontweight='bold')
    ax.text(5.5, 5.5, '五问共用同一判定内核,仅优化维度递进', ha='center', fontsize=10, color='gray')
    ax.text(5.5, 1.2, 'Q1(验证) → Q2(单弹) → Q3(接力) → Q4(协同) → Q5(多对多)',
            ha='center', fontsize=11, fontweight='bold', color='#444441')

    plt.savefig(f'{OUT}/fig_framework.png'); plt.close()

# ===== 修复 fig_flowchart: PSO流程图 =====
def fig_flowchart_v2():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.axis('off')

    steps = [
        (4, 7.0, '初始化\n随机生成粒子群\n(θ, v, t_rel, Δt)', 3.5, 0.9),
        (4, 5.6, '适应度评估\n离散时间步进仿真\n计算有效遮蔽时长 T', 3.5, 0.9),
        (4, 4.2, '更新个体/全局最优\nPSO速度位置更新', 3.5, 0.9),
        (4, 2.8, '收敛判断\n是否达到迭代上限?', 3.5, 0.9),
        (6.5, 1.2, '输出最优解\n(航向、速度、时序)', 2.8, 0.9),
    ]

    for x, y, txt, w, h in steps:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.12",
                               facecolor='#EEEDFE', edgecolor='#534AB7', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, y, txt, ha='center', va='center', fontsize=9.5, color='#3C3489')

    # 主流程箭头
    for y1, y2 in [(6.55, 6.05), (5.15, 4.65), (3.75, 3.25)]:
        ax.annotate('', xy=(4, y2), xytext=(4, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # 收敛分支
    ax.annotate('', xy=(5.8, 2.35), xytext=(5.0, 2.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(5.4, 2.15, '是', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(5.8, 3.25), xytext=(5.0, 3.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(5.4, 3.05, '否', fontsize=10, fontweight='bold')

    ax.text(4, 7.8, 'PSO优化算法流程图', ha='center', fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}/fig_flowchart.png'); plt.close()

if __name__ == '__main__':
    fig_framework_v2(); print('fig_framework ✓')
    fig_flowchart_v2(); print('fig_flowchart ✓')
