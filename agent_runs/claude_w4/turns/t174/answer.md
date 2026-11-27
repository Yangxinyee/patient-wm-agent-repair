The outliers are a handful of NE-only bundles where u never had nearly all the attention in heads 1 to 3; for example, bundle 615 had weights of 0.28, 0.44 and 0.01. Adding the clone there takes weight from the slots.

This is rare: only 0.34% of drug bundles change E by more than 0.05. The clone is also identical in the recorded and edited rollouts, so it cannot bias the dose readouts. I accept it rather than add a key offset.

The clone is therefore essentially exact. What remains is to choose where the surprise term should push the state.

My plan is to use the model's own way of storing new glucose information. At measurement bundles with a finger reading, I regress the step's state update, y_after minus the time-flowed y_before (4×192), on the reading's surprise, observed minus the original model's prediction. That gives a per-mg/dL direction ν_G. I do the same for arterial MAP to get ν_M. Injecting −κ·s_I·ν_G at insulin bundles then makes the model update as if it had learned that glucose is lower.

What would confirm this is a usable direction: a clear surprise coefficient, meaning a substantial partial R² across many state dimensions and stable across slots. If the surprise explains almost nothing of the state update (if the model keeps glucose information mostly in the reading register rather than the state), I will need a different target, such as the value-head input.

```python
import numpy as np, torch
print('F2 cols', list(F2.columns)); print(F2.head(2).to_string())
fa_ = load_original('core.flow.a').double().numpy()
Y3 = X3[:, 192:960].astype(float)
key_b = {(int(s), round(float(h), 4)): i for i, (s, h) in enumerate(zip(sid3, hr3))}
def surprise_dir(event, ctrl_events):
    rows = F2[F2['event'] == event]
    ks, ss, ctr = [], [], []
    for _, r in rows.iterrows():
        k = key_b.get((int(r['stay_id']), round(float(r['hour']), 4)))
        if k is None or k == 0 or sid3[k - 1] != sid3[k]: continue
        ks.append(k); ss.append(float(r['value'] - r[mcol]))
    ks = np.array(ks); ss = np.array(ss)
    dt = (hr3[ks] - hr3[ks - 1]) * 3600.0
    dec = np.exp(-dt[:, None] * np.exp(fa_)[None, :])
    htf_ = Y3[ks - 1].reshape(-1, 4, 192) * dec[:, None, :]
    D = Y3[ks].reshape(-1, 4, 192) - htf_
    D = D.reshape(len(ks), -1)
    others = []
    for ev in ctrl_events:
        sub_ = F2[F2['event'] == ev]
        st_ = set((int(a), round(float(b), 4)) for a, b in zip(sub_['stay_id'], sub_['hour']))
        others.append([((int(sid3[k]), round(float(hr3[k]), 4)) in st_) for k in ks])
    Xr = np.column_stack([np.ones(len(ks)), ss] + [np.array(o, float) for o in others])
    coef, *_ = np.linalg.lstsq(Xr, D, rcond=None)
    resid_full = D - Xr @ coef
    Xr0 = np.delete(Xr, 1, axis=1); c0, *_ = np.linalg.lstsq(Xr0, D, rcond=None); resid0 = D - Xr0 @ c0
    pr2 = 1 - (resid_full**2).sum() / (resid0**2).sum()
    nu = coef[1].reshape(4, 192)
    cs_ = [float(nu[i] @ nu[j] / np.linalg.norm(nu[i]) / np.linalg.norm(nu[j])) for i, j in [(0, 1), (0, 2), (0, 3), (1, 2)]]
    print(f'{event}: n {len(ks)} surprise sd {ss.std():.2f}  |nu| per unit {np.linalg.norm(nu):.4f}  partial R2 (all dims) {pr2:.3f}  '
          f'slot norms {np.round(np.linalg.norm(nu, axis=1), 4)}  slot cos {np.round(cs_, 3)}  |update| typical {np.linalg.norm(D, axis=1).mean():.2f}')
    return nu, ks, ss
nuG, ksG, ssG = surprise_dir('glucose_finger', ['glucose_lab', 'map_arterial', 'map_cuff'])
nuM, ksM, ssM = surprise_dir('map_arterial', ['glucose_finger', 'glucose_lab', 'map_cuff'])
```