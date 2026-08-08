"""
Q5 多弹联合优化: 对每架UAV重新优化θ,v,t_rel,dt共8变量(3弹)
不再固定θ/v为单弹最优,而是做多弹折中
"""
import numpy as np, json, openpyxl
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

def log(msg): print(msg, flush=True)

with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json") as f:
    old = json.load(f)

# 当前分配
assign = {uid: old['details'][uid]['missile'] for uid in old['details']}

def joint_optimize(uav_id, missile_id, init_theta=None, init_v=None):
    """8变量联合优化: θ, v, tr1, df1, tr2, df2, tr3, df3
    目标: 3弹并集最大化
    """
    uav_z = UAVS[uav_id][2]
    if init_theta is None:
        init_theta = np.pi
    if init_v is None:
        init_v = 100.0

    rng = np.random.RandomState(hash(uav_id+missile_id+"multi3")%1000)
    n_p, n_i = 25, 35
    dim = 8
    # θ, v, tr1, df1, tr2, df2, tr3, df3
    lb = np.array([0.0, 70.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    ub = np.array([2*np.pi, 140.0, 12.0, 10.0, 25.0, 10.0, 40.0, 10.0])

    pos = rng.uniform(lb, ub, (n_p, dim))
    pos[0] = [init_theta, init_v, 1.0, 2.0, 5.0, 2.0, 10.0, 3.0]
    vel = np.zeros((n_p, dim))
    pbest = pos.copy()

    def fitness(x):
        th, vv, tr1, df1, tr2, df2, tr3, df3 = x
        if tr2 < tr1 + 1 or tr3 < tr2 + 1: return 0.0
        if uav_z - 4.9*df1**2 < 0 or uav_z - 4.9*df2**2 < 0: return 0.0
        if uav_z - 4.9*df3**2 < 0: return 0.0
        bombs = [(tr1, df1), (tr2, df2), (tr3, df3)]
        T, _, _ = calc_multi_bomb_time(missile_id, uav_id, th, vv, bombs,
                                        dt_step=0.05, n_samples=60)
        return T

    pf = np.array([fitness(p) for p in pos])
    gi = np.argmax(pf); gb, gf = pos[gi].copy(), pf[gi]

    for it in range(n_i):
        r1, r2 = rng.random((n_p, dim)), rng.random((n_p, dim))
        vel = 0.6*vel + 1.5*r1*(pbest-pos) + 1.5*r2*(gb-pos)
        pos = np.clip(pos+vel, lb, ub)
        for i in range(n_p):
            f = fitness(pos[i])
            if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
            if f > gf: gf, gb = f, pos[i].copy()
        if (it+1) % 10 == 0:
            log(f"    iter {it+1}: gbest={gf:.4f}s")

    th, vv, tr1, df1, tr2, df2, tr3, df3 = gb
    bombs = [(tr1, df1), (tr2, df2), (tr3, df3)]
    T_final, singles, _ = calc_multi_bomb_time(missile_id, uav_id, th, vv, bombs,
                                                dt_step=0.01, n_samples=180)
    return (th, vv, bombs, T_final, singles)


log("="*60)
log("Q5 多弹联合优化 (8变量)")
log("="*60)

final = {}
total = 0.0

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid = assign[uid]
    init_th = np.radians(old['details'][uid]['theta_deg'])
    init_v = old['details'][uid]['v']

    log(f"\n--- {uid}→{mid} 联合优化 ---")
    log(f"  初始解: θ={old['details'][uid]['theta_deg']:.2f}°, v={init_v:.1f}")

    th, vv, bombs, T_final, singles = joint_optimize(uid, mid, init_th, init_v)

    # 精确验证每弹
    for j,(tr,df) in enumerate(bombs):
        Ts, _, _, _ = calc_time_precise(mid, uid, th, vv, tr, df)
        log(f"  弹{j+1}: tr={tr:.3f}, df={df:.3f}, T_single={Ts:.4f}s")

    log(f"  并集: {T_final:.4f}s")
    final[uid] = (mid, th, vv, bombs, T_final)
    total += T_final

log(f"\nQ5总遮蔽: {total:.4f}s")

# 写result3.xlsx
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)','干扰的导弹编号'])

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid, th, v, bombs, T_final = final[uid]
    for j,(tr,df) in enumerate(bombs):
        drop = uav_pos(uid, th, v, tr)
        bp, _ = burst_point(uid, th, v, tr, df)
        Ts, _, _, _ = calc_time_precise(mid, uid, th, v, tr, df)
        ws.append([uid, round(np.degrees(th)%360,4), round(v,4), j+1,
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4), round(Ts,4), mid])
    for j in range(len(bombs), 3):
        ws.append([uid, None, None, j+1] + [None]*8)

ws.append([None]*12)
ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
log("\nresult3.xlsx 已更新")
