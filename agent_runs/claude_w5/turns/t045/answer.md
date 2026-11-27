That run failed because `Z` is the loaded npz file, not the array; the states sit inside it, probably under key `x`. I'll pull the array out and rerun the same decoding. The test is unchanged: high cross-validated R² for lagged and residual doses would support a head-level readout edit, and low R² would point to a core-level memory.

```python
print(Z.files)
Xst=Z['x'] if 'x' in Z.files else Z[Z.files[0]]
print(Xst.shape)
Xall=Xst[rows_]
for tg in ['g_last','ins1','ins2','ins3','insres1','ne1','ne2','ne3','neres1']:
    yv=D[tg].values.astype(float)
    r_all=cvr2(Xall,yv,grp); r_out=cvr2(Xall[:,:192],yv,grp); r_h=cvr2(Xall[:,192:960],yv,grp)
    print(tg, 'all %.3f(a=%g)'%r_all, 'out %.3f'%r_out[0], 'h %.3f'%r_h[0])
for nm_ in ['F','L','R','Rres','Nres','Rn','sid','H','Gn','Mg','grid','P','PT','ins','ne','st_sid','st_hr']:
    if nm_ in globals():
        v=globals()[nm_]; print(nm_, type(v).__name__, getattr(v,'shape',None) if not callable(v) else 'func')
```