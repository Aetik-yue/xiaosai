#!/usr/bin/env python3
"""批量修复paper_v4.tex"""
import re

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'r', encoding='utf-8') as f:
    c = f.read()

# 1. 修复小标题格式: 1→一、 2→二、 ...
# 在documentclass后添加ctexset
if '\\ctexset' not in c:
    c = c.replace('\\begin{document}', '''\\ctexset{
  section = {name={、},number=\\chinese{section}},
  subsection = {name={、},number=\\chinese{section}.\\chinese{subsection}},
  subsubsection = {number=\\chinese{section}.\\chinese{subsection}.\\chinese{subsubsection}}
}
\\begin{document}''')

# 2. 补充附录代码（在thebibliography之前）
appendix_code = '''
% ==================== 附录 ====================
\\appendix
\\section{核心代码}

\\subsection{运动学模型 (core.py)}

\\begin{lstlisting}
import numpy as np

G = 9.8
R_SMOKE = 10.0
V_MISSILE = 300.0
V_SINK = 3.0

TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0
FAKE_TARGET = np.array([0.0, 0.0, 0.0])

MISSILES = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0]),
}
UAVS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0]),
}

def missile_pos(mid, t):
    m0 = MISSILES[mid]
    d = (FAKE_TARGET - m0) / np.linalg.norm(FAKE_TARGET - m0)
    return m0 + V_MISSILE * t * d

def uav_pos(uid, theta, v, t):
    u0 = UAVS[uid]
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    return u0 + v * t * d

def burst_point(uid, theta, v, t_rel, dt_fuse):
    t_b = t_rel + dt_fuse
    drop = uav_pos(uid, theta, v, t_rel)
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    bp = drop + v * dt_fuse * d
    bp[2] -= 0.5 * G * dt_fuse ** 2
    return bp, t_b
\\end{lstlisting}

\\subsection{遮蔽判定 (core.py)}

\\begin{lstlisting}
def is_occluded(mid, cloud_pos, t, n_samples=360):
    m_pos = missile_pos(mid, t)
    mc = cloud_pos - m_pos
    d = np.linalg.norm(mc)
    if d <= R_SMOKE:
        return True
    alpha = np.arcsin(R_SMOKE / d)

    phi = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    for p in phi:
        offset = TARGET_RADIUS * np.array([np.cos(p), np.sin(p), 0])
        for center in [TARGET_CENTER, TARGET_CENTER+[0,0,TARGET_HEIGHT]]:
            pt = center + offset
            mp = pt - m_pos
            mp_n = np.linalg.norm(mp)
            if mp_n < 1e-10: continue
            cos_b = np.dot(mp, mc) / (mp_n * d)
            cos_b = np.clip(cos_b, -1, 1)
            beta = np.arccos(cos_b)
            if beta > beta_max:
                beta_max = beta
    return beta_max <= alpha
\\end{lstlisting}

\\subsection{PSO优化器 (optimizer.py)}

\\begin{lstlisting}
def solve_q2(n_particles=30, n_iter=40, seed=42):
    rng = np.random.RandomState(seed)
    lb = np.array([0.0, 70.0, 0.0, 0.0])
    ub = np.array([2*np.pi, 140.0, 10.0, 6.0])
    pos = rng.uniform(lb, ub, (n_particles, 4))
    pos[0] = [np.pi, 120.0, 1.5, 3.6]
    vel = np.zeros((n_particles, 4))
    pbest = pos.copy()

    def fitness(x):
        theta, v, tr, df = x
        if v < 70 or v > 140: return 0.0
        if UAVS['FY1'][2] - 4.9*df**2 < 0: return 0.0
        T, _, _, _ = calc_time_fast('M1', 'FY1', theta, v, tr, df)
        return T

    pf = np.array([fitness(p) for p in pos])
    gi = np.argmax(pf); gb, gf = pos[gi].copy(), pf[gi]

    for it in range(n_iter):
        r1 = rng.random((n_particles, 4))
        r2 = rng.random((n_particles, 4))
        vel = 0.7*vel + 1.5*r1*(pbest-pos) + 1.5*r2*(gb-pos)
        pos = np.clip(pos+vel, lb, ub)
        for i in range(n_particles):
            f = fitness(pos[i])
            if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
            if f > gf: gf, gb = f, pos[i].copy()

    return gb, gf
\\end{lstlisting}
'''

# 在bibliography之前插入附录
c = c.replace('\\begin{thebibliography}', appendix_code + '\n\\begin{thebibliography}')

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'w', encoding='utf-8') as f:
    f.write(c)

print('修复完成')
