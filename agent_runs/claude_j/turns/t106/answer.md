All 8 `last_numeric` inputs carry weight. My best guess is that they mirror `embed.numeric`: [has_last, log-age, then z and has-flag for value, dose and rate]. For a measurement only has_last, age, z_last_value and has_value are nonzero, so only those positions need to be found. All 15 `type_emb` rows are used. `last_status` row 0 is zero, so row 1 is probably added when a previous value exists.

To pin down the remaining unknowns I will build the value-head replica in two stages:
- **First readings of each code in a stay:** these have no "last" register, so their prediction depends only on the type index and the activation. Testing all 15 types times 2 activations against the tool's predicted means should pick out one configuration with a much smaller error than the rest.
- **Readings with a previous value:** I will then search the register layout.

The code goes into `/work/heads.py` so it survives restarts. I use z equal to the prior mean, so a small residual error from the tool's four z draws is expected.

```python
code_txt = r'''
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
'''
open('/work/heads.py', 'w').write(code_txt)
exec(code_txt)
Xd = torch.tensor(S['x'], dtype=torch.float64)
RTb = build_reading_table(S, fa_)
print(RTb.shape, 'matched', RTb['predicted_mean'].notna().sum(), 'first-readings', (RTb.has_last == 0).sum())
prow_ = torch.tensor(RTb.prow.values); dt_ = torch.tensor(RTb.dt.values, dtype=torch.float64)
out_, mh_, gv_, ctx_, zp_ = view_parts(Xd, prow_, dt_)
view0_ = out_ + mh_ + gv_ + Hlin(zp_[:, :32], 'z_head')
codes_ = torch.tensor([CID[e] for e in RTb.event])
cen_ = torch.tensor(np.stack([CEN[e] for e in RTb.event]))
def head_mean(view, typ, act, extra=None, tb=None):
    q = Hln(view + HT('type_emb.weight')[typ], 'query_norm')
    wc = q + HT('embed.code.weight')[codes_]
    if extra is not None: wc = wc + extra
    lg = Hlin(act(Hlin(wc, 'attr_heads.value_bin.0')), 'attr_heads.value_bin.2')
    lg = lg + HT('embed_transition.value_bin.weight')[codes_ * 17 + (tb if tb is not None else 0)]
    p = torch.softmax(lg, -1)
    return (p[:, 1:] * cen_).sum(1), p
m1 = torch.tensor((RTb.has_last == 0).values)
tool_ = torch.tensor(RTb.predicted_mean.values)
ext0 = HT('last_numeric.bias')[None, :] + HT('embed.bins.value.weight')[0] + HT('embed.bins.rate.weight')[0]
resq = []
for an_, act_ in [('gelu', Fn.gelu), ('relu', torch.relu)]:
    for ty in range(15):
        mu_, _ = head_mean(view0_, ty, act_, ext0)
        resq.append((float((mu_ - tool_)[m1].abs().mean()), an_, ty))
resq.sort()
print('best (MAE on first readings):', [(round(a, 3), b, c) for a, b, c in resq[:5]])
print('worst:', round(resq[-1][0], 2))
```