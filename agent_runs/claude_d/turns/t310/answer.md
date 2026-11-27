The notebook settles the target question. Earlier work (runs 46–50) compared power transforms of the NE rate with an overidentification test. The linear form fit best (J=21 on 16 df), and log-linear was clearly rejected (J=582). The cleaned effect kernel (gam_ne_clean) was 3.28, 2.73, 2.02, 1.21 mmHg per unit rate at lags 1–4. So "NE is linear and additive" comes from the records, and the ne_x2 target of about +4.85 on sel1, most of it from doubled rates above 1.0, stands.

E5 extrapolated in the wrong space. It extended the out-of-range atom along an in-range slope taken in the `mlp1` output space. After `mlp2` and the layer norm, that direction is not the one the model's own recorded-rate changes move u along, and it saturates. A more principled alternative is to extrapolate in u-space along the direction recorded rate increases actually move u (du/dr). Then find the `mlp1` output that produces that u.

First I need the in-range u(r) curve for NE START atoms: how big ‖u(r) − u(1.0)‖ is, and how linear u is in r. This run identifies the atom-builder helpers and checks that `aNE(1.0)` reproduces the stored m(1.0). If so, it computes u(r) on a grid and fits du/dr.

If u(0)→u(1) is a displacement of similar size to the one E5 already made (≈2.6) but in a different direction, the u-space approach is promising. If u(r) is a pure step function with no consistent direction, it is not.

```python
import numpy as np
for f_n in [aNE, aNE0, avec, avec2, avec3, atom_vec, aS, f_ne, u_of_s, uenc]:
    co = f_n.__code__; print(f_n.__name__, co.co_varnames[:co.co_argcount], co.co_names[:12])
try:
    a1 = np.asarray(aNE(1.0)).ravel(); print('aNE(1.0) shape', a1.shape)
    mm = np.asarray(m1(a1[None])).ravel(); print('m1(aNE(1)) vs m10 err', float(np.abs(mm-m10).max()))
    rg = np.array([0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.87,0.9,0.95,1.0])
    Mg = np.asarray(m1(np.stack([np.asarray(aNE(r)).ravel() for r in rg])))
    Ug = np.stack([np.asarray(u_from_m(Mg[i])).ravel() for i in range(len(rg))])
    u1 = Ug[-1]
    print('|u(r)-u(1)|', np.round(np.linalg.norm(Ug-u1,axis=1),2))
    print('|m(r)-m(1)|', np.round(np.linalg.norm(Mg-m10,axis=1),2), '|dm|', round(float(np.linalg.norm(dm)),2))
    sel = rg>=0.3; X_ = np.c_[np.ones(sel.sum()), rg[sel]]
    cf_, *_ = np.linalg.lstsq(X_, Ug[sel], rcond=None); dudr = cf_[1]
    res_ = Ug[sel]-X_@cf_; print('dudr norm', round(float(np.linalg.norm(dudr)),2), 'lin R2', round(1-float((res_**2).sum()/((Ug[sel]-Ug[sel].mean(0))**2).sum()),3))
    ue5 = np.asarray(u_from_m(m10+1.0*dm)).ravel()
    print('cos(E5 disp r2, dudr)', round(float((ue5-u1)@dudr/np.linalg.norm(ue5-u1)/np.linalg.norm(dudr)),3))
except Exception as ex: print('ERR', repr(ex))
```