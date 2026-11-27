Run 291 shows that no bin-side combination closes the gap. The recorded-dose mismatch stays at 3.3–3.6 mg/dL and 1.8–2.0 mmHg, so the binning of the hour-12 drug atoms is not the cause. Bins, end atoms, merged bundles and z sampling are now all ruled out.

The remaining pattern points to sensitivity rather than encoding. The tool's per-stay shifts carry 2–5 times more scatter than mine. The recorded-dose prediction is off only for the edited model. Teacher-forced states, which may run at higher precision, matched to 2e-6. My stage-1 edit is ill-conditioned: rank-2 weights with singular values near 3×10⁴ and entries near 1,900, reading a low-variance direction of the normalised drug token by cancelling large terms. If the GPU rollout runs at reduced precision, or perturbs the state slightly, such an edit would turn small errors into large, dose-dependent errors in the decoded dose. The original model would barely notice.

This run measures that sensitivity at the 12:05 step. It compares exact float64 weights with fp16- and bf16-rounded weights, for both the edited and the original model. For each, I check how far the recorded-dose prediction moves and whether rounding moves my local means and recorded-dose predictions toward the tool's. If rounding shifts the edited model by about 3 mg/dL and the original barely, the fix is to rebuild stage 1 with well-conditioned, smaller weights.

```python
import torch
def rounded(Wd, dt):
    o = {}
    for kk, v in Wd.items():
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        o[kk] = t.to(dt).to(torch.float64)
    return o
mx = max(float((v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).abs().max()) for v in WE.values())
print('max |w| in WE:', round(mx, 1))
WV = {'E64': WE, 'Eb16': rounded(WE, torch.bfloat16), 'O64': W64, 'Ob16': rounded(W64, torch.bfloat16)}
if mx < 6e4: WV['E16'] = rounded(WE, torch.float16); WV['O16'] = rounded(W64, torch.float16)
res6, err = [], None
for sid in list(p13.index):
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; globals()['WCUR'] = WE
        st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; rec = {'sid': sid}; dt13 = (13.0 - B[k][0]) * 3600
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = lastval(df, oc, 13.0)
            toks = {nm: enc2(at) for nm, at in (('b', ats), ('x2', scl(ats, drug, 2.0, 1)), ('h', scl(ats, drug, 0.5, 1)))}
            for vn, Wd in WV.items():
                globals()['WCUR'] = Wd
                pv = {}
                for nm, (e2, u2) in toks.items():
                    r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                    pv[nm] = float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
                rec[f'{vn}_{drug[:3]}_base'] = pv['b']; rec[f'{vn}_{drug[:3]}_x2'] = pv['x2'] - pv['b']; rec[f'{vn}_{drug[:3]}_h'] = pv['h'] - pv['b']
        res6.append(rec)
    except Exception as e:
        err = repr(e)[:200]
globals()['WCUR'] = WE
print('n', len(res6), 'err', err)
R6 = pd.DataFrame(res6).set_index('sid')
lines = []
for vn in WV:
    s = vn + ':'
    for dg, tcx, tch, fc in (('ins', 'insulin_x2', 'insulin_half', 'insulin_x2'), ('nor', 'ne_x2', 'ne_half', 'ne_x2')):
        jb = pd.concat([tf13[fc], R6[f'{vn}_{dg}_base']], axis=1).dropna()
        dref = (R6[f'{vn}_{dg}_base'] - R6[f"{'E64' if vn[0]=='E' else 'O64'}_{dg}_base"]).abs().median()
        s += f" {dg}: base mad_tool={float((jb.iloc[:,0]-jb.iloc[:,1]).abs().median()):.2f} |d vs64|={float(dref):.2f} x2={R6[f'{vn}_{dg}_x2'].mean():.2f} h={R6[f'{vn}_{dg}_h'].mean():.2f};"
    lines.append(s)
print('tool means: ins_x2 -4.62 ins_h -0.04 ne_x2 1.20 ne_h -0.50')
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 291: no bin combo fixes base mad (3.3-3.6). Run 292 precision-fragility test at 12:05 step:\n  " + "\n  ".join(lines) + "\n")
```