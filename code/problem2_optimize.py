# -*- coding: utf-8 -*-
"""
A题 问题2：单弹投放策略优化
变量：航向θ, 速度v(70~140), 投放时刻t_r, 引信延时Δt_f
目标：最大化对M1的有效遮蔽时长
方法：粗网格搜索 + 局部精细搜索
"""
import numpy as np
import math
import itertools

G = 9.8
CLOUD_R = 10.0
CLOUD_SINK = 3.0
CLOUD_DUR = 20.0

M1_START = np.array([20000.0, 0.0, 2000.0])
M1_DIST = np.linalg.norm(M1_START)
M1_SPEED = 300.0

FY1_START = np.array([17800.0, 0.0, 1800.0])

TARGET_R = 7.0
TARGET_H = 10.0

def target_points():
    pts = []
    for i in range(8):
        ang = 2 * math.pi * i / 8
        pts.append(np.array([TARGET_R * math.cos(ang), 200.0 + TARGET_R * math.sin(ang), 0.0]))
        pts.append(np.array([TARGET_R * math.cos(ang), 200.0 + TARGET_R * math.sin(ang), TARGET_H]))
    pts.append(np.array([0.0, 200.0, 0.0]))
    pts.append(np.array([0.0, 200.0, TARGET_H]))
    pts.append(np.array([0.0, 200.0, TARGET_H / 2]))
    return pts

TPS = target_points()

def missile(t):
    return M1_START * (1.0 - M1_SPEED * t / M1_DIST)

def compute_obscuration(theta_deg, v, t_r, dt_fuse):
    theta = math.radians(theta_deg)
    dir_vec = np.array([math.cos(theta), math.sin(theta), 0.0])

    # 释放点
    release = FY1_START + dir_vec * v * t_r
    # 起爆点
    t_d = t_r + dt_fuse
    deton = np.array([
        release[0] + dir_vec[0] * v * dt_fuse,
        release[1] + dir_vec[1] * v * dt_fuse,
        release[2] - 0.5 * G * dt_fuse**2
    ])

    t_start = t_d
    t_end = min(t_d + CLOUD_DUR, M1_DIST / M1_SPEED)

    dt = 0.01
    total = 0.0
    in_obs = False

    t = t_start
    while t <= t_end:
        m = missile(t)
        c = np.array([deton[0], deton[1], deton[2] - CLOUD_SINK * (t - t_d)])

        # 角覆盖法
        d_mc = np.linalg.norm(c - m)
        if d_mc <= CLOUD_R:
            obscured = True
        else:
            alpha = math.asin(min(1.0, CLOUD_R / d_mc))
            mc_dir = (c - m) / d_mc
            obscured = True
            for tp in TPS:
                mt = tp - m
                d_mt = np.linalg.norm(mt)
                if d_mt < 1e-9:
                    continue
                cos_a = max(-1.0, min(1.0, np.dot(mc_dir, mt / d_mt)))
                if math.acos(cos_a) > alpha:
                    obscured = False
                    break

        if obscured and not in_obs:
            in_obs = True
        elif not obscured and in_obs:
            in_obs = False

        if obscured:
            total += dt

        t += dt

    return total

# 粗网格搜索
print("=== 问题2 粗网格搜索 ===")
best = (0, None)
count = 0
for theta in range(0, 360, 15):
    for v in [70, 100, 120, 140]:
        for t_r in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            for dt_f in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
                count += 1
                T = compute_obscuration(theta, v, t_r, dt_f)
                if T > best[0]:
                    best = (T, (theta, v, t_r, dt_f))
                    print(f"  新最优: T={T:.3f}s  θ={theta}° v={v} t_r={t_r} Δt={dt_f}")

print(f"\n粗搜索完成 ({count} 组合)")
print(f"最优: T={best[0]:.3f}s  参数: θ={best[1][0]}° v={best[1][1]} t_r={best[1][2]} Δt={best[1][3]}")

# 精细搜索
print("\n=== 精细搜索 ===")
theta0, v0, tr0, df0 = best[1]
best2 = best
for theta in np.arange(max(0, theta0 - 15), min(360, theta0 + 15), 3):
    for v in np.arange(max(70, v0 - 30), min(140, v0 + 30), 5):
        for t_r in np.arange(max(0.1, tr0 - 1.5), tr0 + 1.5, 0.2):
            for dt_f in np.arange(max(0.5, df0 - 3), df0 + 3, 0.3):
                T = compute_obscuration(theta, v, t_r, dt_f)
                if T > best2[0]:
                    best2 = (T, (theta, v, t_r, dt_f))

print(f"精细搜索最优: T={best2[0]:.3f}s")
print(f"参数: θ={best2[1][0]:.1f}° v={best2[1][1]:.1f} t_r={best2[1][2]:.2f} Δt={best2[1][3]:.2f}")

# 计算最优参数下的详细结果
theta, v, t_r, dt_f = best2[1]
theta_rad = math.radians(theta)
dir_vec = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
release = FY1_START + dir_vec * v * t_r
deton = np.array([
    release[0] + dir_vec[0] * v * dt_f,
    release[1] + dir_vec[1] * v * dt_f,
    release[2] - 0.5 * G * dt_f**2
])
print(f"\n释放点: ({release[0]:.2f}, {release[1]:.2f}, {release[2]:.2f})")
print(f"起爆点: ({deton[0]:.2f}, {deton[1]:.2f}, {deton[2]:.2f})")
print(f"起爆时刻: {t_r + dt_f:.2f}s")
