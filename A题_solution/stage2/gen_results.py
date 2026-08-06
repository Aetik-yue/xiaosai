"""生成result1~3.xlsx"""
import numpy as np, openpyxl, json
from core import *

with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\results.json") as f:
    R = json.load(f)

# result1: Q3 FY1 3弹→M1
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)'])
q3th, q3v = np.radians(R['Q3_params'][0]), R['Q3_params'][1]
for j, (tr, df) in enumerate(R['Q3_params'][2]):
    drop = uav_pos('FY1', q3th, q3v, tr)
    bp, _ = burst_point('FY1', q3th, q3v, tr, df)
    from optimizer import calc_time_precise
    Ts, _, _, _ = calc_time_precise('M1','FY1',q3th,q3v,tr,df)
    ws.append([round(R['Q3_params'][0],4), round(q3v,4), j+1,
               round(drop[0],4),round(drop[1],4),round(drop[2],4),
               round(bp[0],4),round(bp[1],4),round(bp[2],4), round(Ts,4)])
ws.append([None]*10)
ws.append(['注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*9)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result1.xlsx")
print("result1.xlsx OK")

# result2: Q4 3机各1弹→M1
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)'])
for uid in ['FY1','FY2','FY3']:
    th, vv, tr, df = R['Q4_params'][uid]
    th_rad = np.radians(th if th > 6.28 else th)  # 已是弧度? 不, json里是弧度
    # Q4_params存的是弧度
    drop = uav_pos(uid, th, vv, tr)
    bp, _ = burst_point(uid, th, vv, tr, df)
    from optimizer import calc_time_precise
    T,_,_,_ = calc_time_precise('M1',uid,th,vv,tr,df)
    ws.append([uid, round(np.degrees(th)%360,4), round(vv,4),
               round(drop[0],4),round(drop[1],4),round(drop[2],4),
               round(bp[0],4),round(bp[1],4),round(bp[2],4), round(T,4)])
ws.append([None]*10)
ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*8)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result2.xlsx")
print("result2.xlsx OK")

# result3: Q5 5机至多3弹→M1M2M3
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)','干扰的导弹编号'])
for uid in ['FY1','FY2','FY3','FY4','FY5']:
    info = R['Q5_params'][uid]
    mid, (th, vv, tr, df), T_val = info['missile'], info['params'], info['T']
    drop = uav_pos(uid, th, vv, tr)
    bp, _ = burst_point(uid, th, vv, tr, df)
    ws.append([uid, round(np.degrees(th)%360,4), round(vv,4), 1,
               round(drop[0],4),round(drop[1],4),round(drop[2],4),
               round(bp[0],4),round(bp[1],4),round(bp[2],4), round(T_val,4), mid])
    for j in range(2):
        ws.append([uid, None, None, j+2]+[None]*8)
ws.append([None]*12)
ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
print("result3.xlsx OK")
