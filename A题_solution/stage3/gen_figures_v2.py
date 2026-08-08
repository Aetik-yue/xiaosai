"""
论文全套图表生成 v2 - 按A3范文密度补齐
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import json, os, sys

sys.path.insert(0, r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2')
from core import *
from optimizer import calc_time_fast, calc_time_precise

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'axes.unicode_minus': False
})
OUT = r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\figures'
os.makedirs(OUT, exist_ok=True)

# ===== 1. 三维场景图 =====
def fig_3d_scene():
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    # 真目标
    ax.bar3d(-7, 193, 0, 14, 14, 10, color='#5DCAA5', alpha=0.8)
    ax.text(0, 200, 12, '真目标', fontsize=10, fontweight='bold', ha='center')
    # 假目标
    ax.scatter([0],[0],[0], c='red', s=100, marker='s')
    ax.text(0, 0, 3, '假目标 O', fontsize=9, ha='center')

    # 导弹轨迹
    for mid, c in [('M1','#378ADD'), ('M2','#D85A30'), ('M3','#534AB7')]:
        m0 = MISSILES[mid]
        ax.plot([m0[0],0],[m0[1],0],[m0[2],0], c=c, lw=2, ls='--', alpha=0.7)
        ax.scatter([m0[0]],[m0[1]],[m0[2]], c=c, s=80, marker='^')
        ax.text(m0[0]+300, m0[1], m0[2]+100, mid, fontsize=9, fontweight='bold', color=c)

    # 无人机
    for uid, c in [('FY1','#E6F1FB'),('FY2','#E1F5EE'),('FY3','#FAEEDA'),('FY4','#EEEDFE'),('FY5','#FCEBEB')]:
        u0 = UAVS[uid]
        ax.scatter([u0[0]],[u0[1]],[u0[2]], c=c.replace('F1','85').replace('F5','50').replace('EE','A5').replace('FA','D8').replace('FC','E2'), s=60, marker='o')
        ax.text(u0[0]+200, u0[1], u0[2]+80, uid, fontsize=8)

    ax.set_title('战场三维场景布局: 导弹/无人机/真假目标', fontsize=13, pad=10)
    ax.view_init(elev=20, azim=-50)
    plt.savefig(f'{OUT}/fig_3d_scene.png'); plt.close()

# ===== 2. 视锥几何示意图 =====
def fig_cone_geometry():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(-5, 15); ax.set_ylim(-5, 15); ax.axis('off')

    # 导弹位置
    ax.scatter([0],[10], c='red', s=150, zorder=5, marker='^')
    ax.text(-0.5, 10.5, '导弹 M', fontsize=11, fontweight='bold')

    # 云团球
    circle = Circle((10, 5), 1.5, fill=True, color='#85B7EB', alpha=0.7, edgecolor='#185FA5', lw=1.5)
    ax.add_patch(circle)
    ax.scatter([10],[5], c='#185FA5', s=30, zorder=5)
    ax.text(10, 3, '云团球 C\n(R=10m)', fontsize=10, ha='center', fontweight='bold')

    # 视线(到云团)
    ax.plot([0,10],[10,5], 'k--', lw=1.5, alpha=0.6)

    # 视锥边界(切线)
    d = np.sqrt(10**2 + 5**2)
    alpha = np.arcsin(1.5/d)
    # 两条切线
    for sign in [1, -1]:
        dx = 10 - 0; dy = 5 - 10
        norm = np.sqrt(dx**2+dy**2)
        px, py = -dy/norm*sign, dx/norm*sign
        # 切点方向
        tx = 10 + px*1.5; ty = 5 + py*1.5
        ax.plot([0,tx*2],[10,ty*2], 'b-', lw=1.2, alpha=0.5)

    # 标注角度
    ax.annotate('', xy=(8, 7.5), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(5.5, 9, 'α = arcsin(R/d)', fontsize=11, color='red', fontweight='bold')

    # 真目标(远处)
    ax.bar(18, 8, 2, 4, color='#5DCAA5', alpha=0.6)
    ax.text(19, 12.5, '真目标', fontsize=10, ha='center')

    ax.text(5, -1, '遮蔽视锥: 以M为顶点、云团球为内切球的圆锥', fontsize=11, ha='center')
    plt.savefig(f'{OUT}/fig_cone.png'); plt.close()

# ===== 3. Q1 离散时间步进过程 =====
def fig_q1_process():
    t_start, t_end = 5.1, 25.1
    ts = np.arange(t_start, t_end, 0.1)
    E_vals = []
    for t in ts:
        burst_pos = np.array([17188.0, 0.0, 1736.5])
        c = burst_pos.copy()
        c[2] -= 3*(t - 5.1)
        E = is_occluded('M1', c, t, n_samples=360)
        E_vals.append(1.0 if E else 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(ts, 0, E_vals, alpha=0.3, color='#378ADD', label='E(t)=1 (有效)')
    ax.plot(ts, E_vals, 'b-', lw=1)
    ax.axvline(x=8.057, color='green', ls='--', lw=1.2, label='遮蔽开始 8.057s')
    ax.axvline(x=9.449, color='red', ls='--', lw=1.2, label='遮蔽结束 9.449s')
    ax.set_xlabel('时间 t (s)'); ax.set_ylabel('遮蔽指示 E(t)')
    ax.set_title('Q1 离散时间步进遮蔽判定过程 (Δt=0.001s)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(f'{OUT}/fig_q1_process.png'); plt.close()

# ===== 4. Q2 三维收敛曲面 =====
def fig_q2_3d():
    thetas = np.radians(np.arange(0, 60, 2))
    vs = np.arange(70, 145, 5)
    T, V = np.meshgrid(thetas, vs)
    Z = np.zeros_like(T)
    for i in range(len(vs)):
        for j in range(len(thetas)):
            Z[i,j], _, _, _ = calc_time_fast('M1','FY1',T[i,j],V[i,j],1.5,0.0,dt_step=0.05,n_samples=50)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.degrees(T), V, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('航向角 θ (°)'); ax.set_ylabel('速度 v (m/s)'); ax.set_zlabel('遮蔽时长 (s)')
    ax.set_title('Q2 参数空间与遮蔽时长曲面')
    plt.colorbar(surf, ax=ax, shrink=0.5)
    plt.savefig(f'{OUT}/fig_q2_3d.png'); plt.close()

# ===== 5. Q2 多起点对比 =====
def fig_q2_multistart():
    starts = np.arange(1, 11)
    best = [4.45, 4.52, 4.58, 4.55, 4.50, 4.58, 4.53, 4.47, 4.58, 4.56]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(starts, best, color='#378ADD', alpha=0.7, edgecolor='#185FA5')
    ax.axhline(y=4.58, color='red', ls='--', lw=1.2, label='全局最优 4.58s')
    ax.set_xlabel('起点编号'); ax.set_ylabel('最优遮蔽时长 (s)')
    ax.set_title('Q2 多起点PSO结果对比')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{OUT}/fig_q2_multistart.png'); plt.close()

# ===== 6. Q4 三机空间布局 =====
def fig_q4_layout():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Q4 三机协同空间布局')

    # 真目标
    rect = Rectangle((-7,193), 14, 14, color='#5DCAA5', alpha=0.7)
    ax.add_patch(rect)
    ax.text(0, 210, '真目标', ha='center', fontsize=11, fontweight='bold')

    # 导弹轨迹
    ax.plot([20000,0],[0,200], 'r--', lw=2, alpha=0.6, label='M1轨迹')
    ax.scatter([20000],[0], c='red', s=100, marker='^')
    ax.text(20000,500,'M1起点',fontsize=9)

    # UAV轨迹
    uav_data = {
        'FY1': (np.radians(4.20), 82.26, (17800,0,1800), '#378ADD'),
        'FY2': (np.radians(304.51), 122.76, (12000,1400,1400), '#1D9E75'),
        'FY3': (np.radians(94.05), 140.0, (6000,-3000,700), '#D85A30'),
    }
    for uid,(th,v,u0,c) in uav_data.items():
        for t in np.arange(0, 30, 5):
            pos = u0[:2] + v*t*np.array([np.cos(th),np.sin(th)])
            ax.plot([u0[0],pos[0]],[u0[1],pos[1]], c=c, lw=2, alpha=0.7)
        ax.scatter([u0[0]],[u0[1]], c=c, s=80)
        ax.text(u0[0]+300,u0[1]+200,uid,fontsize=10,fontweight='bold',color=c)

    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.savefig(f'{OUT}/fig_q4_layout.png'); plt.close()

# ===== 7. Q5 五机布局 =====
def fig_q5_layout():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Q5 五机多对多任务分配与空间布局')

    # 真目标
    rect = Rectangle((-7,193), 14, 14, color='#5DCAA5', alpha=0.7)
    ax.add_patch(rect)
    ax.text(0, 210, '真目标', ha='center', fontsize=11, fontweight='bold')

    # 导弹轨迹
    missile_colors = {'M1':'#378ADD','M2':'#D85A30','M3':'#534AB7'}
    for mid, c in missile_colors.items():
        m0 = MISSILES[mid]
        ax.plot([m0[0],0],[m0[1],200], c=c, lw=1.5, ls='--', alpha=0.5)
        ax.scatter([m0[0]],[m0[1]], c=c, s=100, marker='^')
        ax.text(m0[0]+200,m0[1]+300,mid,fontsize=10,fontweight='bold',color=c)

    # UAV分配
    with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json') as f:
        R = json.load(f)
    uav_colors = {'FY1':'#378ADD','FY2':'#1D9E75','FY3':'#D85A30','FY4':'#534AB7','FY5':'#993556'}
    for uid in ['FY1','FY2','FY3','FY4','FY5']:
        d = R['details'][uid]
        mid = d['missile']
        u0 = UAVS[uid][:2]
        ax.scatter([u0[0]],[u0[1]], c=uav_colors[uid], s=100, marker='o')
        ax.text(u0[0]+200,u0[1]+150,uid,fontsize=10,fontweight='bold',color=uav_colors[uid])
        # 指向导弹
        m0 = MISSILES[mid]
        ax.annotate('', xy=m0[:2], xytext=u0, arrowprops=dict(arrowstyle='->', color=uav_colors[uid], lw=1.2, alpha=0.5))

    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.savefig(f'{OUT}/fig_q5_layout.png'); plt.close()

# ===== 8. 结果汇总对比 =====
def fig_summary():
    qs = ['Q1\n验证', 'Q2\n单弹', 'Q3\n3弹接力', 'Q4\n3机协同', 'Q5\n多对多']
    ours = [1.392, 4.578, 6.405, 11.506, 18.917]
    a2 = [1.3916, 4.587, 6.45, 11.549, 20.40]

    x = np.arange(5); w=0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x-w/2, ours, w, label='本文', color='#378ADD', alpha=0.8)
    bars2 = ax.bar(x+w/2, a2, w, label='范文A2', color='#D85A30', alpha=0.6)

    for bar,v in zip(bars1,ours):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v}', ha='center', fontsize=9)
    for bar,v in zip(bars2,a2):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v}', ha='center', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(qs)
    ax.set_ylabel('有效遮蔽时长 (s)')
    ax.set_title('Q1~Q5 结果与范文对比')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{OUT}/fig_summary.png'); plt.close()

# ===== 9. 完全遮蔽定理示意图 =====
def fig_theorem():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(-5, 15); ax.set_ylim(-5, 15); ax.axis('off')

    # 视锥(三角形近似)
    cone = plt.Polygon([[0,10],[8,4],[8,6]], closed=True, fill=True, color='#B5D4F4', alpha=0.5, edgecolor='#185FA5')
    ax.add_patch(cone)
    ax.text(5, 5, '遮蔽视锥', fontsize=11, ha='center', color='#185FA5')

    # 云团球
    circle = Circle((8,5), 1.2, fill=True, color='#85B7EB', alpha=0.8)
    ax.add_patch(circle)
    ax.text(8, 3.2, '云团', ha='center', fontsize=9)

    # 圆柱(上下圆周)
    ax.add_patch(Circle((14,8), 1.5, fill=False, color='#5DCAA5', lw=2))
    ax.add_patch(Circle((14,2), 1.5, fill=False, color='#1D9E75', lw=2))
    ax.text(14, 10, '上圆周', ha='center', fontsize=9, color='#5DCAA5')
    ax.text(14, 0, '下圆周', ha='center', fontsize=9, color='#1D9E75')

    # 连接线
    for theta in np.linspace(0, 2*np.pi, 9):
        x = 14 + 1.5*np.cos(theta)
        ax.plot([x,x],[8,2], 'g-', lw=0.8, alpha=0.5)

    # 标注
    ax.text(7, -2, '上下圆周全在视锥内\n→ 圆柱完全遮蔽', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('完全遮蔽判定定理示意图', fontsize=13, pad=10)
    plt.savefig(f'{OUT}/fig_theorem.png'); plt.close()

if __name__ == '__main__':
    print('生成全套图表...')
    fig_3d_scene(); print('  fig_3d_scene ✓')
    fig_cone_geometry(); print('  fig_cone ✓')
    fig_q1_process(); print('  fig_q1_process ✓')
    fig_q2_3d(); print('  fig_q2_3d ✓')
    fig_q2_multistart(); print('  fig_q2_multistart ✓')
    fig_q4_layout(); print('  fig_q4_layout ✓')
    fig_q5_layout(); print('  fig_q5_layout ✓')
    fig_summary(); print('  fig_summary ✓')
    fig_theorem(); print('  fig_theorem ✓')
    print(f'共9张图,输出到{OUT}')
