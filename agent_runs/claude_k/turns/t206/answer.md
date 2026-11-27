This result changes the plan. The fwd8 writer put up to about 2 units into z18 and z28, yet the resulting change in the next ctx averages only 0.04 in norm. That compares with a ctx norm of 23 and a natural SD of 0.32 per dimension.

Per unit of norepinephrine, the lag-1 response in ctx is 0.08. Decoding the previous rate from the edited ctx does not improve on the original (R² 0.61 vs 0.61). Only 35% of the ctx change is linear in the lagged doses.

So the quiet z dimensions are quiet because the network barely passes them on: z enters only through z_in into the prelude token, and those columns must be weak. Writing into them cannot carry a dose signal strong enough for a readout without enormous values that would saturate the LayerNorms.

The fix is to also edit how a dedicated z dimension enters the prelude. I would make its z_in column strong, and its posterior output purely my writer. The model hardly uses that dimension, so its loss should cost little factually.

To choose the dimension and the column strength, I need to know:
- the z_in column norms and bias, and the typical z-token size;
- how strongly the active dimensions enter;
- the prelude's parameter layout (how slots attend to the z token);
- whether z_head reads the posterior or only prior draws, which the doc says are prior draws.

This run only inspects weights: it lists those tensors and compares the column norms of z_in and z_head with the natural spread of each z posterior dimension.

```python
sel=[n for n in NAMES if any(s in n for s in ['z_in','z_head','z_post','z_prior','prelude','b_in','c_out','coda','out_norm','delta_in','a_in','core.slots','gap'])]
for n in sel: print(n, SHAPES[n])
Wzi=load_original('z_in.weight') if 'z_in.weight' in NAMES else None
cand=[n for n in NAMES if n.endswith('z_in.weight')]
print('z_in names', cand)
W=np.asarray(load_original(cand[0]),np.float64); bz=np.asarray(load_original(cand[0].replace('weight','bias')),np.float64)
zs=np.r_[S2['x'][:,960:],S5['x'][:,960:]].astype(np.float64)
cn=np.linalg.norm(W,axis=0); sdz=zs.std(0); mz=zs.mean(0)
print('z_in W',W.shape,'bias norm %.2f'%np.linalg.norm(bz))
print('dim: colnorm*sd (effective) sorted:')
o=np.argsort(cn*sdz)
print([(int(j),round(float(cn[j]),3),round(float(sdz[j]),3),round(float(mz[j]),2)) for j in o])
tok=zs@W.T+bz; print('z token norm mean %.2f, sd-per-dim of token %.3f'%(np.linalg.norm(tok,axis=1).mean(), tok.std(0).mean()))
zh=[n for n in NAMES if 'z_head' in n and n.endswith('weight')]; print(zh)
if zh:
    Wh=np.asarray(load_original(zh[0]),np.float64); print('z_head col norms (18,28,27):', np.linalg.norm(Wh,axis=0)[[18,28,27]], 'median', np.median(np.linalg.norm(Wh,axis=0)))
```