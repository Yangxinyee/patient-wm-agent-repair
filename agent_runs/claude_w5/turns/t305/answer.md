`avec` reads its fine bin through the global `fb_`, so I can test bin conventions. There is also a simpler emulation that needs no patching. Nudging a value up by a factor of 1 + 1e-9 makes a value lying exactly on an edge fall in the bin that side='right' would give, and the z-score is unaffected.

I'll redo the hour-13 comparison under three variants:
1. no nudge;
2. nudge only the hour-12 bundle's drug atoms (all doses after the factor);
3. nudge every drug atom in the stay.

I'll compare each with the tool per stay: the factual level, and the x2 and x1/2 shifts.
- **Hypothesis confirmed:** one variant brings the edited model's gap down to the original's level (median about 0.4) and makes the tool's x1/2 ≈ 0 reproducible.
- **Hypothesis refuted:** nothing changes, and bin edges are not the issue.

```python
import numpy as np
def nudge(ats, which=('insulin','norepinephrine')):
    out = []
    for a in ats:
        a2 = dict(a)
        if a.get('kind')==0 and a['code'] in which:
            for k_ in ('rate','dose'):
                if k_ in a2 and a2[k_] is not None: a2[k_] = a2[k_]*(1+1e-9)
        out.append(a2)
    return out
res_n = []
for sid in cmpI.index:
    B = make_bundles(tun[tun.stay_id==sid])
    k = [k for k,(t,ats) in enumerate(B) if 12.0 <= t < 13.0 and any(a.get('kind')==0 and a['code']=='insulin' for a in ats)][0]
    t12, ats = B[k]
    last = [(t, a['value']) for (t, aa) in B[:k+1] for a in aa if a['code']=='glucose_finger' and a.get('value') is not None and a['value']==a['value']]
    tl, xl = last[-1]; dt_s = (13.0-t12)*3600; age_s = (13.0-tl)*3600
    rec = {'sid': sid}
    for var in ['base','P2','P1']:
        for nmW, W in [('E', WE), ('O', W64)]:
            for f in [1.0, 2.0, 0.5]:
                a12 = scale_atoms(ats, 'insulin', f) if f != 1.0 else ats
                pre = [(t, nudge(aa)) for (t, aa) in B[:k]] if var == 'P1' else B[:k]
                a12 = nudge(a12) if var in ('P1','P2') else a12
                rec[f'{var}_{nmW}_{f}'] = pm_(*state_after(W, pre + [(t12, a12)]), dt_s, 'glucose_finger', xl, age_s)
    res_n.append(rec)
RN_ = pd.DataFrame(res_n).set_index('sid').join(cmpI[['tool_fact_e','tool_fact_o','insulin_x2','insulin_half']])
for var in ['base','P2','P1']:
    ge = (RN_.tool_fact_e - RN_[f'{var}_E_1.0']); go = (RN_.tool_fact_o - RN_[f'{var}_O_1.0'])
    sx2 = RN_[f'{var}_E_2.0'] - RN_[f'{var}_E_1.0']; sh = RN_[f'{var}_E_0.5'] - RN_[f'{var}_E_1.0']
    print(var, 'gap_e median|d| %.2f  gap_o %.2f | x2 loc mean %.2f (tool %.2f) corr %.2f | half loc mean %.2f (tool %.2f) corr %.2f' % (ge.abs().median(), go.abs().median(), sx2.mean(), RN_.insulin_x2.mean(), np.corrcoef(sx2, RN_.insulin_x2)[0,1], sh.mean(), RN_.insulin_half.mean(), np.corrcoef(sh, RN_.insulin_half)[0,1]))
```