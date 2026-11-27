I misread the previous output: list position 4 is bundle 5, not bundle 4. Bundle 4 is in fact exact (7e-5). The first real failure is bundle 5, the first bundle with an infusion END atom. Measurement bundles after it are also off, which points at the memory entry that bundle writes.

My hypothesis: END atoms are not treated as actions. They go through the measurement encoder into e_k (and so into memory), and only START atoms go into u_k. That would also explain why the z-posterior mismatches were limited to bundles with END atoms. I'll test it at bundles 5 and 6. Two choices for e_5 (END atoms through `encode`, or the empty encoding) and a few ways of building the END atom vector. If routing END atoms into e is right, both steps should become exact.

```python
evs5 = BUN[st0][5][1]; print([ (a,b) for a,b,_ in evs5])
starts = [ev for ev in evs5 if ev[1]=='start']; ends = [ev for ev in evs5 if ev[1]=='end']
u_start = bvd(starts)[1]; e_emp = bvd(starts)[0]
cands = {}
for kw in [dict(), dict(use_unit=False), dict(use_rate_unit=False), dict(use_unit=False,use_rate_unit=False), dict(end_ekind=0), dict(end_kq=1)]:
    try:
        av = torch.stack([torch.as_tensor(drug_end(ev, **kw)).double() for ev in ends])
        cands[str(kw)] = av
    except Exception as ex_: print('drug_end fail', kw, repr(ex_)[:120])
def enc_e(av):
    try: return torch.as_tensor(encode(av.float(), p='encode')).double()
    except Exception: return torch.as_tensor(encode([a.float() for a in av], p='encode')).double()
def err_k(stay, k, Eover):
    bl_ = BUN[stay]; idx_ = np.where(sid_==stay)[0]
    Es = [bvd(b[1]) for b in bl_]; Es = [Eover.get(j, Es[j]) for j in range(len(Es))]
    times = np.array([b[0] for b in bl_], float)*60.0
    out_p = Xt[idx_[k-1], :192]; h_p = Xt[idx_[k-1], 192:960].view(4,192); z_k = Xt[idx_[k], 960:992]
    js = list(range(0,k+1)); Mem = torch.stack([Es[j][0] for j in js]); ages = torch.tensor(times[k]-times[js], dtype=torch.float64)
    o_, y_ = step(out_p, h_p, float(times[k]-times[k-1]), Es[k][0], Es[k][1], z_k, Mem, ages)
    return round(max((o_-Xt[idx_[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx_[k],192:960]).abs().max().item()),5)
print('current', err_k(st0,5,{}), err_k(st0,6,{}))
for nm, av in cands.items():
    e5 = enc_e(av)
    print(nm, 'H1 e=enc(END),u=enc_act(START):', err_k(st0,5,{5:(e5,u_start)}), err_k(st0,6,{5:(e5,u_start)}))
print('H2b e=empty, u=enc_act(START only):', err_k(st0,5,{5:(e_emp,u_start)}), err_k(st0,6,{5:(e_emp,u_start)}))
```