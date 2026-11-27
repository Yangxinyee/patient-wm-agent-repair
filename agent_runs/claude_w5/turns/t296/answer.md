Two findings. First, `W64`, my "original" float64 weight dict, also contains the kv edit: an earlier copy shared the tensor and was modified in place. So my local "original" baselines all included stage 1. That explains the 1.3e-4 gap from the tool's original. I'll restore it by assigning a fresh copy of the original tensor; `WE` stays edited.

Second, `pred_mean(out, y, dt_s, code, lastx, last_age_s)` computes the head's predicted mean directly from a state, which is exactly what the hour-13 check needs. For every selection-A stay with insulin or norepinephrine at hour 12, I'll teacher-force through the hour-12 drug bundle with factual, x2 and x1/2 doses. I'll then compute the predicted glucose_finger / map_arterial mean at 13:00 for the edited and the true original model.

- If the local hour 13 shows x2 ≈ −3 and x1/2 ≈ 0 as in the tool, the asymmetry lives in the state-to-head map.
- If it shows a symmetric ≈ +1.5 for x1/2, the tool's rollout differs from this setup.

```python
import numpy as np, time
W64[kvn] = load_original(kvn).double()
print('W64 kv orig now:', np.abs(to64(W64[kvn]) - load_original(kvn).double().numpy()).max(), '| WE kv edited:', np.abs(to64(WE[kvn]) - load_original(kvn).double().numpy()).max())
def state_after(W, B):
    r = run_tf(B, step=step_with(W)); return r[-1][0], r[-1][1]
def pm_(o_, y_, dt_s, code, xl, age_s):
    v = pred_mean(o_, y_, dt_s, code, xl, age_s, nz=0); return float(np.asarray(v, float).ravel()[0])
t0 = time.time(); res13 = []
for sid in selA_list:
    B = make_bundles(tun[tun.stay_id==sid])
    ks = [k for k,(t,ats) in enumerate(B) if 12.0 <= t < 13.0 and any(a.get('kind')==0 and a['code'] in ('insulin','norepinephrine') for a in ats)]
    if not ks: continue
    k = ks[0]; t12, ats = B[k]
    for drug, code in [('insulin','glucose_finger'), ('norepinephrine','map_arterial')]:
        if not any(a.get('kind')==0 and a['code']==drug for a in ats): continue
        last = [(t, a['value']) for (t, aa) in B[:k+1] for a in aa if a['code']==code and a.get('value') is not None and a['value']==a['value']]
        if not last: continue
        tl, xl = last[-1]; dt_s = (13.0 - t12)*3600; age_s = (13.0 - tl)*3600
        rec = {'sid': sid, 'drug': drug, 'rate': [a['rate'] for a in ats if a.get('kind')==0 and a['code']==drug][0]}
        for nmW, W in [('E', WE), ('O', W64)]:
            v_ = {f: pm_(*state_after(W, B[:k] + [(t12, scale_atoms(ats, drug, f) if f != 1.0 else ats)]), dt_s, code, xl, age_s) for f in [1.0, 2.0, 0.5]}
            rec[nmW+'f'] = v_[1.0]; rec[nmW+'x2'] = v_[2.0]-v_[1.0]; rec[nmW+'half'] = v_[0.5]-v_[1.0]
        res13.append(rec)
D13 = pd.DataFrame(res13); print('n', len(D13), 'time %.0f' % (time.time()-t0))
print(D13.groupby('drug')[['rate','Ex2','Ehalf','Ox2','Ohalf']].mean().round(3))
print('edited-orig factual at h13:', D13.groupby('drug').apply(lambda g: (g.Ef-g.Of).mean()).round(3).to_dict())
```