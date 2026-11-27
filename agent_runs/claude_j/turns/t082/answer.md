There's a dtype mismatch: some tensors are float32 and some float64. I'll print the dtypes, then run the whole replica step in float64 (weights, states, encodings and ages), which also makes the comparison more precise. Expected result unchanged: one variant at about 1e-5.

```python
print(Wt['core.flow.a'].dtype, Xt.dtype, bundle_vecs(bl[1][1])[0].dtype)
W = {k_: torch.as_tensor(v_).double() for k_,v_ in Wt.items()}
Xt = torch.tensor(x).double()
_bv = bundle_vecs
def bvd(evs):
    e_, u_ = _bv(evs); return e_.double(), u_.double()
def run_variant(mode, age_unit, mem_cur, kmax=12, stay=None):
    stay = st0 if stay is None else stay
    bl_ = BUN[stay]; idx_ = np.where(sid_==stay)[0]
    errs = []
    Es = [bvd(b[1]) for b in bl_]
    times = np.array([b[0] for b in bl_], float)*60.0
    for k in range(1, min(kmax, len(bl_))):
        out_p = Xt[idx_[k-1], :192]; h_p = Xt[idx_[k-1], 192:960].view(4,192); z_k = Xt[idx_[k], 960:992]
        e_k, u_k = Es[k]
        js = list(range(max(0,k-128+(1 if mem_cur else 0)), k+(1 if mem_cur else 0)))
        if len(js)==0: js=[k]
        Mem = torch.stack([Es[j][0] for j in js]); ages = torch.tensor((times[k]-times[js])/age_unit, dtype=torch.float64)
        o_, y_ = step(out_p, h_p, float(times[k]-times[k-1]), e_k, u_k, z_k, Mem, ages, mode)
        errs.append(max((o_-Xt[idx_[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx_[k],192:960]).abs().max().item()))
    return np.round(errs, 5)
for mode in ['post','pre']:
    for au in [1.0, 60.0, 3600.0]:
        for mc in [True, False]:
            try: print(mode, au, mc, run_variant(mode, au, mc))
            except Exception as ex_: print(mode, au, mc, 'ERR', repr(ex_)[:200])
```