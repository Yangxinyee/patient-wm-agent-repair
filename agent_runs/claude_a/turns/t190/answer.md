My prelude function reproduces `tf_loop`'s stored E and attention weights exactly (error 0). So all prelude-stage quantities can now be computed for any bundle set.

For the 3-dims-per-drug design, the gate units go in the first block's FFN, so I also need block internals. That means self-attention over slots and cross-attention to the bundle memory M with age embeddings. To build M I need the bundle layout in `ds`: the bundle table, the per-step row lists, and whether e is zero at drug-only bundles. I'll inspect those briefly before writing the block step.

```python
bd=ds['bd']; print(bd.shape, bd.columns.tolist()); print(bd.head(4).to_string()[:500])
sr=ds['srows']; print('srows', type(sr).__name__, len(sr), [len(x) for x in sr[:5]], sr[0][:5] if len(sr) else '')
print('R keys', list(ds['R'].keys())[:10]); print('rd cols', ds['rd'].columns.tolist())
e_=ds['e']; u_=ds['u']; nz_e=(e_.abs().sum(1)>0); nz_u=(u_.abs().sum(1)>0)
print('bundles with e=0: %d, u!=0: %d, both: %d, of %d'%(int((~nz_e).sum()), int(nz_u.sum()), int(((~nz_e)&nz_u).sum()), len(e_)))
```