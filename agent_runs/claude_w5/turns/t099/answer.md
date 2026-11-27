Run 94 is a real step forward. With an empty-set encoding in place of a zero vector for u, including the gap term now helps instead of hurting. The best error drops to about 0.48 (ctx = out + LN(slot mean) + gap, without the time flow). So the empty-set u is probably correct, and my earlier conclusions about the gap and the flow were distorted by that error.

With this fixed, I'll redo a joint search over the choices I tested earlier in isolation:
- time flow on or off;
- gap input scaling;
- unit indices for glucose and MAP;
- bin-edge side.

A combination with an error near zero would complete the rebuild. If the best stays well above zero, another hidden piece remains.

```python
res_j = []
EE_cache = {}
for gi_, mi_ in [(0,0)] + list(itertools.permutations([2,3,4,5], 2)):
    for side in ['left','right']:
        opt_ = {'unit': {'glucose_finger':gi_, 'glucose_lab':gi_, 'map_arterial':mi_, 'map_cuff':mi_}, 'rate_unit': {}, 'side': side}
        EE_cache[(gi_, mi_, side)] = torch.stack([setenc_a([atom_vec(a, opt_) for a in wa], 'encode', Fnn.gelu) for wa in wa_list])
GG_m10 = torch.stack([G_in(d_, 'min10') for d_ in dts_])
for key_, EEv in EE_cache.items():
    for fl_nm, Hv in [('flow', HF_), ('noflow', HN_)]:
        for g_nm, Gv_ in [('s10', GG_s10), ('min10', GG_m10), ('nogap', torch.zeros_like(GG_s10))]:
            zu = mlp_(torch.cat([O_ + Hv + Gv_, EEv, U_empty[None].expand_as(EEv)], 1), 'z_post', 'gelu')[:, :32]
            res_j.append((round(float(torch.median(torch.norm(zu-ZT_, dim=1)/torch.norm(ZT_, dim=1))),4), key_, fl_nm, g_nm))
res_j.sort(key=lambda t: t[0]); print(res_j[:12])
```