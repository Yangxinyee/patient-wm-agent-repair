The best variant so far: the current bundle is in memory, action-only bundles also enter memory, and age is in raw seconds. That gives out-error 0.052 against out rms 1.0. Including action-only bundles helped, so the content of their memory entry probably matters. I'll test three options:
- e+u;
- the encoder applied to an empty set (a constant rather than zero);
- u for action-only bundles.

```python
empty_e = ln_(mlp_(lin_(torch.tensor([0.0], dtype=torch.float64), 'encode.size', P0), 'encode.mlp2', P0), 'encode.norm', P0)
def step_err3(mmode, stays_=stays12[:4], u_empty_tok=True):
    errs = []; oerr = []
    for s_ in stays_:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        mins = [g_.minute.iloc[0] for g_ in bl_]; enc = [enc_bundle(g_) for g_ in bl_]
        def mvec(j):
            e_, u_, ha, hm = enc[j]
            if mmode == 'e+u': return e_ + u_
            if mmode == 'empty_const': return e_ if hm else empty_e
            if mmode == 'u_if_act': return e_ if hm else u_
            return e_
        for k_ in range(1, len(bl_)):
            prev = X4d[idx_[k_-1]]; h_p = prev[192:960].reshape(4,192)
            e_, u_, ha, hm = enc[k_]
            if mmode == 'empty_const' and not hm: e_ = empty_e
            js = list(range(k_+1))[-128:]
            Mv = torch.stack([mvec(j) for j in js])
            Mage = torch.tensor([(mins[k_]-mins[j])*60.0 for j in js], dtype=torch.float64)
            y = step(h_p, e_, u_, X4d[idx_[k_], 960:992], (mins[k_]-mins[k_-1])*60.0, Mv, Mage, P0, {})
            errs.append(((y - X4d[idx_[k_], 192:960].reshape(4,192))**2).mean().sqrt().item())
            oerr.append(((ln_(y.mean(0), 'core.out_norm', P0) - X4d[idx_[k_], :192])**2).mean().sqrt().item())
    return round(float(np.mean(errs)), 3), round(float(np.mean(oerr)), 4)
for mm_ in ['e', 'e+u', 'empty_const', 'u_if_act']:
    print(mm_, step_err3(mm_))
```