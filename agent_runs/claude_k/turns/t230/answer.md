Run 213 shows the encoder helpers are still in the kernel:
- `setenc_my(V, pre)` encodes a set of atom vectors;
- `mk_best(event, value)` builds an observation atom;
- `u_rec(ri, rn)` builds the action encoding for a drug bundle;
- `E_list`, `E_EMPTY` and `U_EMPTY` are cached.

Next I'll find the training-record DataFrame and what `E_list` holds. Then I can rebuild the full z posterior input `[ctx, e, u]` for drug bundles. If that rebuild matches the stored z posterior, I can fit the predicted-rate decoder on exactly the inputs z_post sees. A poor match would mean my e or u rebuild is off and needs fixing first.

```python
dfs=[(k,v.shape) for k,v in globals().items() if isinstance(v,pd.DataFrame) and not k.startswith('_')]
print(sorted(dfs,key=lambda t:-t[1][0])[:8])
print(type(E_list).__name__, len(E_list), type(E_list[0]).__name__, np.shape(E_list[0]), 'E_EMPTY', np.shape(E_EMPTY), 'U_EMPTY', np.shape(U_EMPTY))
dH=S2['has_insulin']; dN=S2['has_norepinephrine']; print('S2 drug bundles: ins',int(dH.sum()),'ne',int(dN.sum()),'either',int(((dH+dN)>0).sum()),'of',len(dH))
```