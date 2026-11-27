The nudge test is informative. With right-side bins for the scaled hour-12 atoms, my local x2 shift at hour 13 becomes −3.10, matching the tool's −3.08 exactly, and the x1/2 correlation rises to 0.66. So bins on exact edges do matter to my edit, much more than to the original, and the tool may bin rescaled atoms differently.

But the factual-level gap (median 2.1) is unchanged, so something else in the rollout's encoding of the hour-12 drug atoms also differs. My stage 1 amplifies u by about 3e4 along its read direction, so a change the original ignores could produce it.

Next I'll test plausible rollout-side re-encodings of the hour-12 bundle, comparing the factual level and the x2/x1/2 shifts with the tool:
- P3: norepinephrine dose set to its rate, as "rate × 1 hour" without weight;
- P4: dose attributes dropped;
- P5: END atoms dropped;
- each combined with right-side bins.

If one variant closes the factual gap to about 0.4, that is what the rollout does.

```python
import numpy as np
def var_atoms(ats, mode):
    out = []
    for a in ats:
        if mode == 'P5' and a.get('kind')==2: continue
        a2 = dict(a)
        if a.get('kind')==0 and a['code'] in ('insulin','norepinephrine'):
            if mode == 'P3' and a['code']=='norepinephrine' and 'rate' in a2: a2['dose'] = a2['rate']
            if mode == 'P4': a2.pop('dose', None)
            for k_ in ('rate','dose'):
                if k_ in a2 and a2[k_] is not None: a2[k_] = a2[k_]*(1+1e-9)
        out.append(a2)
    return out
res_v = []
for sid in cmpI.index:
    B = make_bundles(tun[tun.stay_id==sid])
    k = [k for k,(t,ats) in enumerate(B) if 12.0 <= t < 13.0 and any(a.get('kind')==0 and a['code']=='insulin' for a in ats)][0]
    t12, ats = B[k]
    last = [(t, a['value']) for (t, aa) in B[:k+1] for a in aa if a['code']=='glucose_finger' and a.get('value') is not None and a['value']==a['value']]
    tl, xl = last[-1]; dt_s = (13.0-t12)*3600; age_s = (13.0-tl)*3600
    rec = {'sid': sid}
    for var in ['P3','P4','P5']:
        for nmW, W in [('E', WE), ('O', W64)]:
            for f in [1.0, 2.0, 0.5]:
                a12 = var_atoms(scale_atoms(ats, 'insulin', f) if f != 1.0 else ats, var)
                rec[f'{var}_{nmW}_{f}'] = pm_(*state_after(W, B[:k] + [(t12, a12)]), dt_s, 'glucose_finger', xl, age_s)
    res_v.append(rec)
RV_ = pd.DataFrame(res_v).set_index('sid').join(cmpI[['tool_fact_e','tool_fact_o','insulin_x2','insulin_half']])
for var in ['P3','P4','P5']:
    ge = (RV_.tool_fact_e - RV_[f'{var}_E_1.0']); go = (RV_.tool_fact_o - RV_[f'{var}_O_1.0'])
    sx2 = RV_[f'{var}_E_2.0'] - RV_[f'{var}_E_1.0']; sh = RV_[f'{var}_E_0.5'] - RV_[f'{var}_E_1.0']
    print(var, 'gap_e median|d| %.2f  gap_o %.2f | x2 loc %.2f (tool %.2f) corr %.2f | half loc %.2f (tool %.2f) corr %.2f' % (ge.abs().median(), go.abs().median(), sx2.mean(), RV_.insulin_x2.mean(), np.corrcoef(sx2, RV_.insulin_x2)[0,1], sh.mean(), RV_.insulin_half.mean(), np.corrcoef(sh, RV_.insulin_half)[0,1]))
```