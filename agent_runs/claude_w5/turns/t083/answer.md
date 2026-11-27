Run 78 isolates the problem to the gap term. Removing gap(log dt/10) entirely raises the correlation with the recorded z from 0.79 to 0.89. So my version of the gap network adds a wrong signal, while the other ctx terms behave as expected: dropping e or the normalised slot mean makes things much worse. The set-encoder variants make no real difference, so the encoder as documented is probably fine.

The gap network is Linear(1→192), an activation, then Linear(192→192); the activation is not specified. A time-embedding layer like this often uses a periodic or saturating activation (sin, tanh, SiLU) instead of GELU. I'll test those activations, each with and without the time flow on h. The right combination should give a clear jump toward an exact match; if none does, the gap input scaling is the next suspect.

```python
GACT = {'gelu': Fnn.gelu, 'relu': Fnn.relu, 'silu': Fnn.silu, 'tanh': torch.tanh, 'sin': torch.sin, 'cos': torch.cos,
        'id': lambda x: x, 'sigmoid': torch.sigmoid, 'softplus': Fnn.softplus}
def Gv(dt, act, scale='log_s_10'):
    x_ = {'log_s_10': math.log(dt)/10, 'log_s': math.log(dt), 'log1p_s_10': math.log1p(dt)/10, 'log_min_10': math.log(dt/60)/10,
          'log_h': math.log(dt/3600), 'log_h1p': math.log1p(dt/3600)}[scale]
    return lin_(GACT[act](lin_(torch.tensor([x_]), 'gap.0')), 'gap.2')
res_g = []
for ga in GACT:
    for fl in ['rate_s', 'none']:
        r_ = zp_generic(stays_t, lambda o,h,dt,wa, ga=ga, fl=fl: torch.cat([o + ln_(flow_variant(h, dt, fl).mean(0),'core.out_norm') + Gv(dt, ga), E_(wa), U0]), max_n=80)
        res_g.append((r_, ga, fl))
res_g.sort(key=lambda t: t[0][0]); print(res_g[:8])
```