38 of the sel1 stays (which have saved states) also appear in the fwd_3 factual output. I can therefore validate a local reimplementation of the value head on those stays without spending a forward pass.

My previous local functions were lost in a restart, so this time I will write them to /work/lib.py.

The steps are: compute the view from out, the time-decayed h, gap and z_head(z) with z drawn from the prior. Then compute q = query_norm(view + type_emb), add the code embedding and last-value register, and run the value MLP plus transition table.

I will compare the predicted means against the tool's. A correlation near 1 with small mean absolute differences would confirm the reconstruction. A poor match would mean one of my guesses is wrong: activation, gap scaling, type index or register features.

```python
lib = r'''
import torch, numpy as np
torch.set_grad_enabled(False)
_cache = {}
def T(n):
    return torch.as_tensor(np.asarray(load(n)), dtype=torch.float64)
def ln(x, name, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), T(name+'.weight'), T(name+'.bias'), eps=eps)
def lin(x, name, bias=True):
    y = x @ T(name+'.weight').T
    return y + T(name+'.bias') if bias else y
ACT = {'gelu': torch.nn.functional.gelu, 'relu': torch.relu, 'silu': torch.nn.functional.silu, 'tanh': torch.tanh}
def mlp2(x, name, act='gelu'):
    return lin(ACT[act](lin(x, name+'.0')), name+'.2')
def ctx_from_state(out, h4, dt_s, gap_div=10.0, act='gelu'):
    dec = torch.exp(-dt_s[:,None,None] * torch.exp(T('core.flow.a'))[None,None,:])
    hd = h4 * dec
    g = mlp2(torch.log(dt_s)[:,None]/gap_div, 'gap', act)
    return out + ln(hd.mean(1), 'core.out_norm') + g
def z_draws(ctx, nz, gen, act='gelu'):
    o = mlp2(ctx, 'z_prior', act); mu, lsd = o[:, :32], o[:, 32:]
    eps = torch.randn((nz,)+mu.shape, generator=gen, dtype=torch.float64)
    return mu[None] + torch.exp(lsd)[None]*eps
'''
open('/work/lib.py','w').write(lib); exec(lib)
fa = T('core.flow.a'); print('flow tau (s) range:', float(torch.exp(-fa).min()), float(torch.exp(-fa).max()))
for nm in ['embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','last_status.weight','embed.code.weight']:
    W = T(nm); print(nm, 'row norms:', np.round(W.norm(dim=1).numpy(),2)[:17])
print('last_numeric col norms', np.round(T('last_numeric.weight').norm(dim=0).numpy(),2))
print('code ids:', [(c['id'], c['name']) for c in VOCAB['codes']])
# states rows
St = pd.DataFrame({'stay': z['stay_id'], 'hour': z['hour'], 'ins': z['has_insulin'], 'ne': z['has_norepinephrine'], 'n': z['n_atoms']})
print(St.head(12).to_string())
```