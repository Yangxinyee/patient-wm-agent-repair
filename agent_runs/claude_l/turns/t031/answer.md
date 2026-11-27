The embedding tables show which indices are actually used. Rows that were never trained keep their initialisation norm of about 0.27, and padding rows are zero.

- **Constant every time:** kq always uses index 0. text, status, route, action and discharged_to always sit at their zero padding row, so they contribute nothing.
- **kind:** WORLD and END are used, REVEAL is not.
- **unit:** four values are used (indices 2–5). These must be mg/dL, mmHg and the two dose units, in an order I still need to determine.
- **rate_unit:** two values are used (2 and 3).

The first-bundle state is identical across stays (spread 1.5e-5), so it depends only on the weights.

I'll start the re-implementation from that first bundle (hospital_admission at −2; initial h = 0, out = 0, empty memory). The z posterior mean depends only on the atom embedding, the set encoder, the gap net and z_post. I'll try the plausible variants (activation GELU vs SiLU vs ReLU; first-bundle dt of 0 or 1 s; whether kq[0] is added) and see which one reproduces the tool's z exactly. An exact match confirms those conventions. If nothing matches, an assumption about the input layout is wrong.

```python
import torch, torch.nn.functional as F
torch.set_grad_enabled(False)
Wt = {n: load(n) for n in NAMES}
def lin(x, n, bias=True):
    y = x @ Wt[n+'.weight'].T
    if bias and (n+'.bias') in Wt: y = y + Wt[n+'.bias']
    return y
def ln(x, n, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), Wt[n+'.weight'], Wt[n+'.bias'], eps)
ACTS = {'gelu': F.gelu, 'silu': F.silu, 'relu': F.relu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh')}
def mlp(x, n, act):  # n.0, n.2
    return lin(act(lin(x, n+'.0')), n+'.2')
def atom_vec(code, kind=0, ekind=0, unit=0, rate_unit=0, vb=0, db=0, rb=0, num=(0,0,0,0,0,0), use_kq=True):
    x = Wt['embed.code.weight'][code] + Wt['embed.kind.weight'][kind] + Wt['embed.ekind.weight'][ekind]
    if use_kq: x = x + Wt['embed.kq.weight'][0]
    x = x + Wt['embed.categorical.unit.weight'][unit] + Wt['embed.categorical.rate_unit.weight'][rate_unit]
    x = x + Wt['embed.bins.value.weight'][vb] + Wt['embed.bins.dose.weight'][db] + Wt['embed.bins.rate.weight'][rb]
    x = x + Wt['embed.numeric.weight'] @ torch.tensor(num, dtype=torch.float32) + Wt['embed.numeric.bias']
    return x
def set_encode(atoms, pre, act):
    n = atoms.shape[0]
    s = mlp(atoms, pre+'.mlp1', act).sum(0) / np.sqrt(n) + lin(torch.tensor([np.log(1+n)], dtype=torch.float32), pre+'.size')
    return ln(mlp(s, pre+'.mlp2', act), pre+'.norm')
z_true = torch.tensor(X[0, 960:992]); out_true = torch.tensor(X[0,:192])
best = []
for act_name, act in ACTS.items():
    for use_kq in [True, False]:
        e1 = set_encode(atom_vec(11, 0, 3, use_kq=use_kq)[None], 'encode', act)
        for gap_in in [0.0, np.log(1)/10, -1.0, None]:
            g = torch.zeros(192) if gap_in is None else mlp(torch.tensor([gap_in], dtype=torch.float32), 'gap', act)
            ctx = Wt['core.out_norm.bias'] + g
            zp = mlp(torch.cat([ctx, e1, torch.zeros(192)]), 'z_post', act)[:32]
            best.append(((zp - z_true).abs().max().item(), act_name, use_kq, gap_in))
best.sort(key=lambda t: t[0]); print(best[:8])
```