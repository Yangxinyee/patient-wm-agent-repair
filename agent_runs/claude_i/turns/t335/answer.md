The variable list was cut off, but it shows what I need:
- `x` (5156×576) is very likely the z-posterior input [ctx, e, u] for every bundle of the fwd_3 train stays.
- `zp` / `zm` look like the matching z outputs.
- `X3`, `S3`, `H3` are the tool states with stay and hour.
- `hi_` / `hn_` are probably the insulin and norepinephrine bundle flags.
- `TR` holds the kernel-implied true shifts for the 100 calibration stays (`sel1`).

Before building decoders on these, I'll verify the identities:
- `zpost_mean` on `x` should reproduce `zp`.
- The flags should count about 1362 insulin bundles.
- I'll look at how hours and drug rows are stored, so I can attach each insulin bundle's rate and decision glucose.

```python
xt=torch.as_tensor(x,dtype=torch.float32)
zz_=zpost_mean(xt[:,:192],xt[:,192:384],xt[:,384:]).numpy()
print('zpost(x) vs zp maxerr', float(np.abs(zz_-np.asarray(zp)[:, :32]).max()), 'median', float(np.median(np.abs(zz_-np.asarray(zp)[:, :32]))))
for nm in ['hi_','hn_','nat','ok','S3','H3','sid']:
    v=np.asarray(globals()[nm]); print(nm, v.dtype, v[:4], 'sum' if v.dtype==bool else 'uniq', (int(v.sum()) if v.dtype==bool else len(np.unique(v))))
print(train.event.unique()[:12]); print(train[train.event.str.contains('insulin')].head(2).to_string()[:400])
```