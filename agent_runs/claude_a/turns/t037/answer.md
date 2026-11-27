The rebuilt heads do not match yet. The best variant still misses the reported predicted means by 11 mg/dL RMSE for glucose and 5.4 mmHg for MAP. Two things did come out. Row 0 of `last_status` is zero, so the status term does not matter for readings. The time-flow constants are in seconds, with timescales from 600 s to 6.4e7 s, so my decay is right.

A mismatch this large points to a wrong discrete choice rather than the noise from the z draws. The likeliest suspect is the `type_emb` index: I assumed it equals the code id. Next I will search the type index over all 15 values for glucose_finger and map_arterial separately, crossed with activation, register layout and age units. I will also check how large the z prior spread and the unexplained `embed.kq` and `embed.text` tables are. If some type index brings the error near zero, that fixes the mapping. If none does, the problem is elsewhere, perhaps in an unlisted query term such as `embed.kq`.

```python
print('kq row norms', P['embed.kq.weight'].norm(dim=1).numpy().round(3), 'text', P['embed.text.weight'].norm(dim=1).numpy().round(3))
print('type_emb norms', P['type_emb.weight'].norm(dim=1).numpy().round(2))
gm = (R.event=='glucose_finger').values; mm = (R.event=='map_arterial').values
best = {}
for evn, msk in [('glucose_finger', gm), ('map_arterial', mm)]:
    out_ = []
    for typ in range(15):
        for act, layout, age_s in itertools.product(['relu','gelu','silu'], ['L1','L2','L4','L5'], [True, False]):
            off = typ - codes_  # so that codes_+off = typ
            m_, p0 = head_mean(act, dts_opts['s'], 10.0, layout, age_s, None, off, False)
            e_ = (m_ - obs_pm)[torch.tensor(msk)]
            out_.append((e_.pow(2).mean().sqrt().item(), typ, act, layout, age_s))
    out_.sort()
    best[evn] = out_[:6]
    print(evn, out_[:6])
# prior spread
A = F.relu
hdec = HS[kprev] * torch.exp(-dts_opts['s'][:, None, None] * torch.exp(P['core.flow.a']))
ctx = OUT[kprev] + ln(hdec.mean(1), 'core.out_norm') + mlp((torch.log(dts_opts['s'])/10)[:, None], 'gap', A)
pr = mlp(ctx, 'z_prior', A)
print('prior logsd mean', pr[:, 32:].mean().item(), 'sd', torch.exp(pr[:, 32:]).mean().item(), 'z_head norm', P['z_head.weight'].norm().item())
```