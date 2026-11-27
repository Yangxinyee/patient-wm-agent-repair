Ages are in seconds, and with all bundles in memory the first three steps match exactly. The first mismatch is the step right after a drug-only bundle: at that point the memory holds the drug-only bundle's entry. So the question is what a drug-only bundle writes into memory. I'll try four candidates for that entry: the empty-set encoding, zeros, the action vector u, and e + u. I'll also try leaving it out. The right choice should make every step match.

```python
def run_variant3(memfill, kmax=14, stay=None):
    stay = st0 if stay is None else stay
    bl_ = BUN[stay]; idx_ = np.where(sid_==stay)[0]
    Es = [bvd(b[1]) for b in bl_]; times = np.array([b[0] for b in bl_], float)*60.0
    actonly = [all(ev[0] in ('insulin','norepinephrine') for ev in b[1]) for b in bl_]
    def memvec(j):
        e_, u_ = Es[j]
        if not actonly[j]: return e_
        return {'e':e_, 'zero':torch.zeros_like(e_), 'u':u_, 'e+u':e_+u_}[memfill]
    errs = []
    for k in range(1, min(kmax, len(bl_))):
        out_p = Xt[idx_[k-1], :192]; h_p = Xt[idx_[k-1], 192:960].view(4,192); z_k = Xt[idx_[k], 960:992]
        js = [j for j in range(0,k+1) if (memfill!='skip' or not actonly[j])][-128:]
        if memfill=='skip' and actonly[k]: js = js + [k]
        Mem = torch.stack([memvec(j) if memfill!='skip' else Es[j][0] for j in js]); ages = torch.tensor((times[k]-times[js]), dtype=torch.float64)
        o_, y_ = step(out_p, h_p, float(times[k]-times[k-1]), Es[k][0], Es[k][1], z_k, Mem, ages)
        errs.append(max((o_-Xt[idx_[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx_[k],192:960]).abs().max().item()))
    return np.round(errs, 5)
print('e3 norm', Es_n := bvd(BUN[st0][3][1])[0].norm().item(), 'u3 norm', bvd(BUN[st0][3][1])[1].norm().item(), BUN[st0][3][1][0][:2])
for mf in ['e','zero','u','e+u','skip']:
    print(mf, run_variant3(mf))
```