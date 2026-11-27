Run 286 shows the bundle layout. The hour-12.083 drug bundle can hold two insulin atoms:
- kind 2: the end of the previous hour's record, carrying the previous rate;
- kind 0: the start of the new record.

My local `scale_atoms` scales only kind-0 atoms. The tool's documented edit scales every drug atom from hour 12 on, which includes the end atoms. My decoders may use the end atom's rate (the previous rate) to estimate c_hat. If so, scaling it in the tool would cancel much of the response and could reverse it for stays where the rate just went down.

Stay 2020000016 fits this: its bundle ends 3.0 and starts 1.0, and both edits push glucose down in the tool.

This run computes per-stay local h13 shifts with the current edited weights under two variants: V0 (scale only kind-0 atoms) and V1 (scale all atoms of the drug). It then correlates each with the tool's per-stay h13 shifts from fwd_9. If V1 matches the tool (high correlation, similar means, including the wrong-sign ×½ at r = 1.0 stays), the cause is confirmed. The fix would then be to refit the decoders on V1-scaled data.

```python
import numpy as np, torch, pandas as pd
for n in edited():
    WE[n] = torch.from_numpy(np.asarray(load(n), dtype=np.float64))
WCUR = WE
def stepW(out, h, dt, e, u, Mv, Ma):
    return mystep(dict(out0=out, h0=h, dt=dt, e=e, u=u, Mv=Mv, Ma=Ma))
def scl(ats, drug, f, allk):
    o = []
    for a in ats:
        a2 = dict(a)
        if a2['code'] == drug and (allk or a2.get('kind') == 0):
            a2['rate'] = a2['rate'] * f; a2['dose'] = a2['dose'] * f
        o.append(a2)
    return o
def lastval(df, code, h):
    r = df[(df.event == code) & (df.hour < h)]
    return (float(r.value.iloc[-1]), float((h - r.hour.iloc[-1]) * 3600)) if len(r) else (float('nan'), float('nan'))
res, err = [], None
for sid in p13.index:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; rec = {'sid': sid}
        dt13 = (13.0 - B[k][0]) * 3600
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = lastval(df, oc, 13.0)
            def pm(at):
                e2, u2 = enc2(at)
                r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                return float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
            base = pm(ats)
            for allk in (0, 1):
                for f, nm in ((2.0, 'x2'), (0.5, 'h')):
                    rec[f'{drug[:3]}_{nm}_V{allk}'] = pm(scl(ats, drug, f, allk)) - base
        res.append(rec)
    except Exception as e:
        err = repr(e)[:200]
print('n', len(res), 'last err', err)
Rl = pd.DataFrame(res).set_index('sid')
lines = []
for tc, lc in (('insulin_x2','ins_x2'), ('insulin_half','ins_h'), ('ne_x2','nor_x2'), ('ne_half','nor_h')):
    for v in ('V0', 'V1'):
        c = f'{lc}_{v}'
        if c not in Rl: continue
        j = pd.concat([p13[tc], Rl[c]], axis=1).dropna()
        lines.append(f"{tc} {v}: n={len(j)} corr={j.corr().iloc[0,1]:.3f} tool={j.iloc[:,0].mean():.2f} local={j.iloc[:,1].mean():.2f}")
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 287: bundles hold kind0 (start) + kind2 (END, prev rate) atoms; my scale_atoms scaled only kind0. Local h13 vs tool (fwd_9):\n  " + "\n  ".join(lines) + "\n")
```