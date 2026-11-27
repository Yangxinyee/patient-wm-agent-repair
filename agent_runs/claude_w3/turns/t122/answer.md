The flow convention is right: exp(a) is a decay rate per second, with timescales from 594 s to 6.5·10⁷ s. The embedding row usage is also consistent:
- kq row 0, kind rows 0 and 2, ekind rows 1–4, unit rows 2–5 and rate_unit rows 2–3 are trained.
- Visits fall on the zero padding row.

So the inputs look correct. The remaining 0.10 could simply be the GPU running in reduced precision (bf16). I'll test this by rerunning my computation with weights and activations rounded to bf16. If that alone shifts the output by about 0.1, my reimplementation is as exact as it can get.

```python
def bf(x): return x.to(torch.bfloat16).to(torch.float64)
Pbf = {k: bf(v) for k, v in P0.items()}
def lin_b(x, p, Pp):
    y = bf(bf(x) @ Pp[p+'.weight'].T)
    return bf(y + Pp[p+'.bias']) if (p+'.bias') in Pp else y
def mlp_b(x, p, Pp): return lin_b(bf(F_.gelu(lin_b(x, p+'.0', Pp))), p+'.2', Pp)
diffs = []; derr = []
for s_ in stays12[:6]:
    idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
    for k_ in range(1, len(bl_)):
        g_ = bl_[k_]
        if g_.code.isin([8,9]).any(): continue
        vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, {}) for r in g_.itertuples()]
        e_ = set_enc_(vecs, 'encode', P0); u_ = torch.zeros(192, dtype=torch.float64)
        prev = X4d[idx_[k_-1]]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
        dt_ = (g_.minute.iloc[0] - bl_[k_-1].minute.iloc[0])*60.0
        hf = h_p*torch.exp(-dt_*torch.exp(P0['core.flow.a']))
        ctx_ = out_p + ln_(hf.mean(0), 'core.out_norm', P0) + mlp_(torch.tensor([np.log(dt_)/10]), 'gap', P0)
        z64 = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0)[:32]
        zb16 = mlp_b(torch.cat([ctx_, e_, u_]), 'z_post', Pbf)[:32]
        diffs.append(((z64-zb16)**2).mean().sqrt().item()); derr.append(((z64 - X4d[idx_[k_], 960:992])**2).mean().sqrt().item())
print('bf16 vs f64 rms', np.mean(diffs), ' f64 vs dump', np.mean(derr))
```