Bundle 3, a drug-only bundle, matches exactly only when its memory entry is the empty-set encoding. So that part is right. Bundle 4 (a cuff MAP reading 55 minutes later) is still off by 0.9 under every memory variant. That suggests the problem is not the memory entry but the timing: the dt used for the time flow, or the ages, may be measured from a different reference after a drug-only bundle. Next I'll vary only the dt used for decay at bundle 4 (3300 s vs 3600 s vs others) and the age reference. If one choice gives zero error, the clock is the issue. If none does, the cause is something else about steps that follow a drug bundle.

```python
def one_step_err(stay, k, dt_override=None, age_shift=None, mem_js=None):
    bl_ = BUN[stay]; idx_ = np.where(sid_==stay)[0]
    Es = [bvd(b[1]) for b in bl_]; times = np.array([b[0] for b in bl_], float)*60.0
    out_p = Xt[idx_[k-1], :192]; h_p = Xt[idx_[k-1], 192:960].view(4,192); z_k = Xt[idx_[k], 960:992]
    js = list(range(0,k+1)) if mem_js is None else mem_js
    ages = (times[k]-times[js]).astype(float)
    if age_shift is not None: ages = ages + age_shift(np.array(js), k)
    Mem = torch.stack([Es[j][0] for j in js])
    dt = float(times[k]-times[k-1]) if dt_override is None else dt_override
    o_, y_ = step(out_p, h_p, dt, Es[k][0], Es[k][1], z_k, Mem, torch.tensor(ages, dtype=torch.float64))
    return round(max((o_-Xt[idx_[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx_[k],192:960]).abs().max().item()),5)
print([b[0] for b in BUN[st0][:8]])
for dto in [None, 3600.0, 3000.0, 300.0, 0.0, 7200.0]:
    print('dt', dto, one_step_err(st0, 4, dt_override=dto))
```