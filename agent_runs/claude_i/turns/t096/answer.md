GELU with gap input log(dt in seconds)/10 is clearly best. The z error is 0.061 on measurement-only bundles and 0.102 on action-only bundles, but that is still far from exact against a z spread of 0.26. Other parts of the input must be wrong.

Measurement-only bundles avoid the drug encoder, so I'll search there first. The grid covers:
- the unit indices for mg/dL and mmHg (12 ordered pairs);
- the kq index (0-3);
- whether the empty action set is zero or the encoded empty set;
- whether ctx uses h before or after the time flow.

One combination giving error near 1e-4 would confirm the measurement path exactly.

```python
def ctx_of2(gscale, act, flow=True):
    hp = torch.zeros(NB, 4, 192, dtype=torch.float64); op = torch.zeros(NB, 192, dtype=torch.float64)
    hp[1:] = h3[:-1]; op[1:] = out3[:-1]
    hf = hp * torch.exp(-dt_s[:, None, None] * torch.exp(Wt('core.flow.a'))) if flow else hp
    return op + LN(hf.mean(1), (192,), Wt('core.out_norm.weight'), Wt('core.out_norm.bias'), eps=1e-5) + mlp_(gscale(dt_s.clamp(min=1))[:, None], 'gap', act)
results_ = []
ctxF = ctx_of2(gs['ls10'], 'gelu', True); ctxN = ctx_of2(gs['ls10'], 'gelu', False)
units_try = [(a_, b_) for a_ in range(2, 6) for b_ in range(2, 6) if a_ != b_]
for kq in range(4):
    for (ug, um) in units_try:
        cfg = {'unit': {'glucose_finger': ug, 'glucose_lab': ug, 'map_arterial': um, 'map_cuff': um, 'insulin': 4, 'norepinephrine': 5}, 'runit': {'insulin': 2, 'norepinephrine': 3}, 'end_attrs': True, 'kq': kq}
        for emp in ['zero', 'enc']:
            enc_ = encode_bundles(A3, cfg, 'gelu', empty=emp)
            mo = prev_ok & (enc_['nu'] == 0)
            for fl_, cx_ in [('flow', ctxF), ('noflow', ctxN)]:
                zp = mlp_(torch.cat([cx_, enc_['e'], enc_['u']], 1), 'z_post', 'gelu')[:, :32]
                results_.append((round((zp - zm3).abs().mean(1)[mo].mean().item(), 4), kq, ug, um, emp, fl_))
results_.sort()
print(results_[:8])
```