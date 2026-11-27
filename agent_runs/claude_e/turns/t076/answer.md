The measurement type index is 1. The remaining error is systematic and correlates strongly with the last-value z: −0.63 for glucose and −0.73 for cuff MAP. So the last-value register inputs are placed in the wrong `last_numeric` columns or scaled wrongly. A smaller part is noise from the tool's 4 z-draws.

Next I will precompute q once and brute-force the column assignment of has-last, log-age (in seconds, minutes or hours), last z and a possible has-value flag. If one assignment cuts the MAE sharply and removes the correlation with last z, the register is identified.

```python
import itertools, time
def precompute_q(R, nz=8, seed=0, type_idx=1):
    gen = torch.Generator().manual_seed(seed)
    prev = torch.as_tensor(R.prev.values); xs = X[prev]
    out = xs[:, :192]; h4 = xs[:, 192:960].reshape(-1,4,192)
    dt = torch.as_tensor(R.dt.values, dtype=torch.float64)
    ctx = ctx_from_state(out, h4, dt)
    zs = z_draws(ctx, nz, gen)
    view = ctx[None] + lin(zs, 'z_head')
    return ln(view + T('type_emb.weight')[type_idx], 'query_norm')
Q8 = precompute_q(R)
code_t = torch.as_tensor(R.code.values); lvb_t = torch.as_tensor(R.lvb.values)
CONST = T('embed.code.weight')[code_t] + T('embed.bins.value.weight')[lvb_t]
TRANS = T('embed_transition.value_bin.weight')[code_t*17 + lvb_t]
CENTM = torch.stack([CENT[int(c)] for c in R.code.values])
W0 = T('attr_heads.value_bin.0.weight'); b0 = T('attr_heads.value_bin.0.bias'); W2 = T('attr_heads.value_bin.2.weight'); b2 = T('attr_heads.value_bin.2.bias')
LNW = T('last_numeric.weight'); LNB = T('last_numeric.bias')
def head_from_f(f, Q=Q8):
    wc = Q + (CONST + f @ LNW.T + LNB)[None]
    p = torch.softmax(torch.nn.functional.gelu(wc @ W0.T + b0) @ W2.T + b2 + TRANS[None], -1).mean(0)
    return (p[:,1:]*CENTM).sum(1).numpy()
has = torch.as_tensor(R.has.values); zl = torch.as_tensor(R.zl.values); age = torch.as_tensor(R.age.values)
las = {'s': torch.log1p(age)/10, 'm': torch.log1p(age/60)/10, 'h': torch.log1p(age/3600)/10}
best=[]; t0=time.time()
for au, la in las.items():
    for ih, il, iz in itertools.permutations(range(8), 3):
        for ihv in [None] + [c for c in range(8) if c not in (ih,il,iz)]:
            f = torch.zeros(len(R), 8, dtype=torch.float64)
            f[:, ih] = has; f[:, il] = la; f[:, iz] = zl
            if ihv is not None: f[:, ihv] = has
            pm = head_from_f(f)
            best.append((np.abs(pm - R.pm.values).mean(), au, ih, il, iz, ihv))
best.sort(key=lambda x: x[0]); print('time', round(time.time()-t0,1))
for b in best[:12]: print(b)
```