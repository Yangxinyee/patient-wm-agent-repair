The recurrent state is large (h rms 66) and my one-step error is about 4.8–6 rms. Including the current bundle in memory gives the best result, but none of the variants is exact.

Next I'll measure the error in the normalised summary `out`, which is what the heads read, so it is easier to interpret. I'll also vary the age convention (units, and whether log(1+age) is divided by 10) with the current bundle in memory.

```python
def step_err2(var, stays_=stays12[:4]):
    errs = []; oerr = []
    for s_ in stays_:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        mins = [g_.minute.iloc[0] for g_ in bl_]; enc = [enc_bundle(g_) for g_ in bl_]
        for k_ in range(1, len(bl_)):
            prev = X4d[idx_[k_-1]]; h_p = prev[192:960].reshape(4,192)
            e_, u_, ha, hm = enc[k_]
            js = [j for j in range(k_+1) if (enc[j][3] or var.get('mem_all'))][-128:]
            Mv = torch.stack([enc[j][0] for j in js])
            Mage = torch.tensor([(mins[k_]-mins[j])*60.0/var.get('au',1.0) for j in js], dtype=torch.float64)
            var2 = dict(var); var2['age_s'] = var.get('as_', 1.0)
            y = step(h_p, e_, u_, X4d[idx_[k_], 960:992], (mins[k_]-mins[k_-1])*60.0, Mv, Mage, P0, var2)
            errs.append(((y - X4d[idx_[k_], 192:960].reshape(4,192))**2).mean().sqrt().item())
            oerr.append(((ln_(y.mean(0), 'core.out_norm', P0) - X4d[idx_[k_], :192])**2).mean().sqrt().item())
    return round(float(np.mean(errs)), 3), round(float(np.mean(oerr)), 4)
for var in [dict(), dict(as_=10.0), dict(au=60.0), dict(au=3600.0), dict(au=3600.0, as_=10.0), dict(mem_all=True)]:
    print(var, step_err2(var))
```