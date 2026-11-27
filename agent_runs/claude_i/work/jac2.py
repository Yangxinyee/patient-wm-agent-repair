
import numpy as np, torch
def build_info(recs, sid3, hr3, NB):
    R = recs.copy(); R['tm'] = np.round(R['hour'].values*60).astype(int)
    grp = {k: g for k, g in R.groupby(['stay_id','tm'])}
    info = []
    for k in range(NB):
        t = int(round(float(hr3[k])*60)); g = grp.get((int(sid3[k]), t)); d = {'t': t, 'sid': int(sid3[k])}
        if g is not None:
            for ev, val, rate in zip(g['event'], g['value'], g['rate']):
                if ev in ('insulin','norepinephrine'):
                    if rate == rate: d[ev] = float(rate)
                elif ev in ('glucose_finger','glucose_lab','map_arterial','map_cuff'): d[ev] = float(val)
        info.append(d)
    return info
def build_reg(info, codes=('glucose_finger','glucose_lab','map_arterial','map_cuff')):
    NB = len(info); LV = {c: np.full(NB, np.nan) for c in codes}; LT = {c: np.full(NB, np.nan) for c in codes}
    for c in codes:
        cur_s, v, t = None, np.nan, np.nan
        for k in range(NB):
            if info[k]['sid'] != cur_s: cur_s, v, t = info[k]['sid'], np.nan, np.nan
            LV[c][k], LT[c][k] = v, t
            if c in info[k]: v, t = info[k][c], info[k]['t']
    return LV, LT
