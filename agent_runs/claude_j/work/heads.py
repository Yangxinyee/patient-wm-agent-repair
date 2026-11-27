
import torch, numpy as np, pandas as pd, math
import torch.nn.functional as Fn
def HT(n): return load(n).double()
def Hlin(x, p): return x @ HT(p + '.weight').T + HT(p + '.bias')
def Hln(x, p, eps=1e-5): return Fn.layer_norm(x, (x.shape[-1],), HT(p + '.weight'), HT(p + '.bias'), eps)
NUMV = {(d_['code'], d_['attribute']): d_ for d_ in VOCAB['numeric']}
CID = {c_['name']: c_['id'] for c_ in VOCAB['codes']}
CEN = {k: np.array(v, float) for k, v in VOCAB['value_bin_centres'].items()}
def zs_(code, x):
    d_ = NUMV[(code, 'value')]; return float(np.clip((x - d_['median']) / d_['iqr'], -10, 10))
def fb_(code, x):
    d_ = NUMV[(code, 'value')]; return 1 + int(np.sum(np.array(d_['fine_bin_edges']) < x))
MEAS = ['glucose_finger', 'glucose_lab', 'map_arterial', 'map_cuff']
def build_reading_table(Sz, fa):
    sidS = np.asarray(Sz['stay_id']); rows = {}
    for r, s in enumerate(sidS): rows.setdefault(int(s), []).append(r)
    recs = []
    for s, rl in rows.items():
        bl = BUN[s]; last = {}
        for b, (mnt, evs) in enumerate(bl):
            for (ev, kind, row) in evs:
                if ev in MEAS and kind == 'world' and b >= 1:
                    L = last.get(ev)
                    recs.append(dict(stay_id=s, minute=mnt, b=b, prow=rl[b - 1], dt=(mnt - bl[b - 1][0]) * 60.0, event=ev,
                                     observed=row.value, has_last=int(L is not None),
                                     age=(mnt - L[0]) * 60.0 if L else 0.0, last_val=L[1] if L else np.nan))
            for (ev, kind, row) in evs:
                if ev in MEAS and kind == 'world': last[ev] = (mnt, row.value)
    RT_ = pd.DataFrame(recs)
    fa2 = fa.copy(); fa2['minute'] = np.round(fa2['hour'] * 60).astype(int)
    return RT_.merge(fa2[['stay_id', 'minute', 'event', 'predicted_mean', 'nll']], on=['stay_id', 'minute', 'event'], how='left')
def view_parts(Xd, prow, dt, gap_scale=10.0):
    out = Xd[prow, 0:192]; h = Xd[prow, 192:960].reshape(-1, 4, 192)
    hf = h * torch.exp(-dt[:, None, None] * torch.exp(HT('core.flow.a')))[:, :, :]
    mh = Hln(hf.mean(1), 'core.out_norm')
    gv = Hlin(Fn.gelu(Hlin(torch.log(dt)[:, None] / gap_scale, 'gap.0')), 'gap.2')
    ctx = out + mh + gv
    zp = Hlin(Fn.gelu(Hlin(ctx, 'z_prior.0')), 'z_prior.2')
    return out, mh, gv, ctx, zp
