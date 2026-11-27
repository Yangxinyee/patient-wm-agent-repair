The bundles line up with the states hour by hour. The `NameError` arose because the previous run stopped before it defined `reading_table`. I will load `/work/prep.py` and build the training reading table. I will then compute q, the value-head query, for every reading with the validated head replica, using z at the prior mean. That gives the regressors for fitting the state readouts.

```python
exec(open('/work/prep.py').read())
RT4 = reading_table(S4, BUN4)
print(RT4.shape, RT4.event.value_counts().to_dict())
Xd4 = torch.tensor(S4['x'], dtype=torch.float64)
def q_for(Xz, RTz, zmode='mean', seed=0):
    prow = torch.tensor(RTz.prow.values); dt = torch.tensor(RTz.dt.values, dtype=torch.float64)
    out, mh, gv, ctx, zp = view_parts(Xz, prow, dt)
    if zmode == 'mean':
        z = zp[:, :32]
    else:
        gen = torch.Generator().manual_seed(seed)
        z = zp[:, :32] + torch.exp(zp[:, 32:]) * torch.randn(zp[:, :32].shape, generator=gen, dtype=torch.float64)
    view = out + mh + gv + Hlin(z, 'z_head')
    return Hln(view + HT('type_emb.weight')[1], 'query_norm')
q4 = q_for(Xd4, RT4)
print('q4', tuple(q4.shape), 'mean |q|', round(float(q4.abs().mean()), 3))
```