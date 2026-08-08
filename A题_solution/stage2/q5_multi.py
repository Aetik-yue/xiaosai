"""
Q5 多弹搜索: 对每架UAV逐弹搜索第2、3弹(固定theta,v)
如实报告结果,每架机填满3行
"""
import numpy as np, json, openpyxl
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

def log(msg): print(msg, flush=True)

with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json") as f:
    R = json.load(f)

# 当前分配
details = R['details']

final_bombs = {}
total_union = 0.0

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    d = details[uid]
    mid = d['missile']
    th = np.radians(d['theta_deg'])
    v = d['v']
    tr1 = d['t_rel']
    df1 = d['dt_fuse']
    T1 = d['T']

    log(f"\n{'='*50}")
    log(f"{uid}→{mid} | θ={d['theta_deg']:.2f}°, v={v:.1f}, 弹1: T={T1:.3f}s")
    log(f"{'='*50}")

    bombs = [(tr1, df1)]
    T_current = T1

    # 搜索第2弹
    log(f"  搜索第2弹 (t_rel≥{tr1+1:.1f}s)...")
    best_union_2 = T_current
    best_2 = None

    for df2 in np.arange(0, 10.5, 1.0):
        if UAVS[uid][2] - 4.9*df2**2 < 0: break
        for tr2 in np.arange(tr1+1, 50, 1.0):
            T_u, singles, _ = calc_multi_bomb_time(mid, uid, th, v,
                [(tr1, df1), (tr2, df2)], dt_step=0.05, n_samples=60)
            if T_u > best_union_2 + 0.05:  # 有显著增量
                best_union_2 = T_u
                best_2 = (tr2, df2, T_u)

    if best_2:
        tr2, df2, Tu2 = best_2
        # PSO精修
        rng = np.random.RandomState(hash(uid+"2")%1000)
        n_p, n_i = 20, 25
        lb = np.array([max(tr1+1, tr2-2), max(0, df2-1.5)])
        ub = np.array([tr2+2, df2+1.5])
        pos = rng.uniform(lb, ub, (n_p, 2))
        pos[0] = [tr2, df2]
        vel2 = np.zeros((n_p, 2))
        pb2 = pos.copy()
        def fit2(x):
            tr, df = x
            if tr<tr1+1: return 0.0
            if UAVS[uid][2]-0.5*G*df**2<0: return 0.0
            Tu,_,_ = calc_multi_bomb_time(mid,uid,th,v,[(tr1,df1),(tr,df)],dt_step=0.02,n_samples=80)
            return Tu
        pf2 = np.array([fit2(p) for p in pos])
        gi = np.argmax(pf2); gb,gf = pos[gi].copy(),pf2[gi]
        for it in range(n_i):
            r1,r2 = rng.random((n_p,2)),rng.random((n_p,2))
            vel2 = 0.6*vel2+1.8*r1*(pb2-pos)+1.8*r2*(gb-pos)
            pos = np.clip(pos+vel2, lb, ub)
            for i in range(n_p):
                f = fit2(pos[i])
                if f>pf2[i]: pf2[i],pb2[i]=f,pos[i].copy()
                if f>gf: gf,gb=f,pos[i].copy()
        tr2,df2 = gb
        Tu2, s2, _ = calc_multi_bomb_time(mid,uid,th,v,[(tr1,df1),(tr2,df2)],dt_step=0.01,n_samples=180)
        log(f"    第2弹: tr={tr2:.3f}, df={df2:.3f}, 并集={Tu2:.3f}s, 增量={Tu2-T_current:.3f}s")
        bombs.append((tr2, df2))
        T_current = Tu2
    else:
        log(f"    第2弹无增量 (最佳并集={best_union_2:.3f}s)")

    # 搜索第3弹
    last_tr = bombs[-1][0]
    log(f"  搜索第3弹 (t_rel≥{last_tr+1:.1f}s)...")
    best_union_3 = T_current
    best_3 = None

    for df3 in np.arange(0, 10.5, 1.0):
        if UAVS[uid][2] - 4.9*df3**2 < 0: break
        for tr3 in np.arange(last_tr+1, 55, 1.0):
            all_b = bombs + [(tr3, df3)]
            T_u, _, _ = calc_multi_bomb_time(mid, uid, th, v, all_b, dt_step=0.05, n_samples=60)
            if T_u > best_union_3 + 0.05:
                best_union_3 = T_u
                best_3 = (tr3, df3, T_u)

    if best_3:
        tr3, df3, Tu3 = best_3
        bombs.append((tr3, df3))
        T_current = Tu3
        log(f"    第3弹: tr={tr3:.3f}, df={df3:.3f}, 并集={Tu3:.3f}s")
    else:
        log(f"    第3弹无增量 (最佳并集={best_union_3:.3f}s)")

    # 精确验证最终并集
    T_final, singles, _ = calc_multi_bomb_time(mid, uid, th, v, bombs, dt_step=0.005, n_samples=240)
    total_union += T_final
    final_bombs[uid] = (mid, th, v, bombs, T_final)
    log(f"  → 最终{len(bombs)}弹, 并集={T_final:.3f}s")

log(f"\n{'='*50}")
log(f"Q5总遮蔽(并集求和): {total_union:.3f}s")
log(f"{'='*50}")

# 写result3.xlsx
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)','干扰的导弹编号'])

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid, th, v, bombs, T_final = final_bombs[uid]
    for j,(tr,df) in enumerate(bombs):
        drop = uav_pos(uid, th, v, tr)
        bp, _ = burst_point(uid, th, v, tr, df)
        # 单弹遮蔽时长
        Ts, _, _, _ = calc_time_precise(mid, uid, th, v, tr, df) if tr>0 else (0,[],np.zeros(3),0)
        ws.append([uid, round(np.degrees(th)%360,4), round(v,4), j+1,
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4), round(Ts,4), mid])
    # 补空行至3弹
    for j in range(len(bombs), 3):
        ws.append([uid, None, None, j+1] + [None]*8)

ws.append([None]*12)
ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
log("result3.xlsx 已更新")
