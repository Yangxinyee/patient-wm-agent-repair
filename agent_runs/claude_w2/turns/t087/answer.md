My helper functions are still in the kernel: `stay_atoms(df_stay)`, `atom_vec`, `encode`, `zpred`, `ln`, `mlp`, `lin`, `zsc` and `fbin`. `stay_atoms` expects a stay's DataFrame (I passed an id by mistake). The train records `trs` are indexed by stay_id.

To build the local core step I need three things:
- the bundle and atom construction, exactly as the model sees it;
- the codes and kinds tables;
- the layout of the fwd_3 states file, to line my bundles up with the tool's.

So this run calls `stay_atoms` on one train stay of selT, prints the vocab code and kind tables, and lists the arrays in the states file. If my bundle count for that stay matches the tool's, the bundling is right. Then I can write the core step and check it against the stored h, out and z one bundle at a time.

```python
print('codes', VOCAB['codes']); print('event_kinds', VOCAB['event_kinds']); print('atom_kinds', VOCAB['atom_kinds'])
sid_ = selT[0]
A_ = stay_atoms(trs.loc[[sid_]] if not isinstance(trs.loc[sid_], pd.Series) else trs.loc[[sid_]])
print(type(A_), len(A_) if hasattr(A_, '__len__') else '')
if isinstance(A_, (list, tuple)): print(A_[:4])
elif hasattr(A_, 'head'): print(A_.head(12).to_string()); print(A_.columns.tolist())
print('S files', {k_: S[k_].shape for k_ in S.files})
for k_ in S.files:
    if S[k_].ndim == 1: print(k_, S[k_][:12])
m0 = (S['stay'] if 'stay' in S.files else st_ids) == sid_
print('tool bundles for stay', sid_, int(m0.sum()), 'hours', np.round((S['hour'] if 'hour' in S.files else hrs)[m0][:14], 3))
```