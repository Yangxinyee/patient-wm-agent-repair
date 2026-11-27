Each bundle carries its encoded measurement set `e`, its encoded action set `u` (192-d), counts and labels. So I can build training data for a u-token dose decoder directly. For about 150 train stays I'll encode the bundles with insulin and NE scaled by 0.5, 1, 2 and 2.5, which covers the range the ×2 edits reach. The targets are the actual (scaled) insulin dose and NE rate, with zero for empty-action bundles. I'll fit linear decoders on prelude head 2's 48-d value output `V2·norm1(u)`. If R² is about 0.99 or higher and the decoders stay accurate at 2–2.5× doses, a rank-1 addition to head 2's output projection can inject an exact dose signal without disturbing the head's original function.

```python
nw_, nb_ = Wo['core.prelude.0.norm1.weight'], Wo['core.prelude.0.norm1.bias']
V2_ = Wo['core.prelude.0.attn.kv.weight'][192+96:192+144]
ln_ = lambda x, w, b: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * w + b
U_rows, T_rows, F_rows = [], [], []
for s_k in keys3[:150]:
    df0 = G3[s_k]
    for drug_, fs in [('insulin', [0.5, 1.0, 2.0, 2.5]), ('norepinephrine', [0.5, 2.0, 2.5])]:
        for f_ in fs:
            df2 = df0.copy(); m2_ = df2['event'] == drug_; df2.loc[m2_, ['rate', 'dose']] *= f_
            st_ = df2[df2['rate'].notna()]
            for b in bundles_of_stay(df2, RU0):
                if b['nA'] == 0 and not (drug_ == 'insulin' and f_ == 1.0): continue
                at = st_[np.isclose(st_['hour'], b['t'])]
                di = at.loc[at['event'] == 'insulin', 'rate'].sum(); dn = at.loc[at['event'] == 'norepinephrine', 'rate'].sum()
                U_rows.append(b['u']); T_rows.append((di, dn)); F_rows.append(f_ if b['nA'] else 0.0)
U_ = np.array(U_rows); T_ = np.array(T_rows); Fv = np.array(F_rows)
O2 = ln_(U_, nw_, nb_) @ V2_.T
print("n", len(U_), "frac empty", np.round((Fv == 0).mean(), 3), "max ins", T_[:,0].max(), "max ne", np.round(T_[:,1].max(),2))
for j_, nm_ in enumerate(['ins', 'ne']):
    A_ = np.c_[O2, np.ones(len(O2))]; cdec = np.linalg.lstsq(A_, T_[:, j_], rcond=None)[0]; pr_ = A_ @ cdec
    r2_ = 1 - ((pr_ - T_[:, j_])**2).mean() / T_[:, j_].var()
    byf = [np.round(np.sqrt(((pr_ - T_[:, j_])[Fv == f_]**2).mean()), 3) for f_ in [0.0, 0.5, 1.0, 2.0, 2.5]]
    print(nm_, "R2 from head-2 values", np.round(r2_, 4), "rmse by factor [empty,.5,1,2,2.5]", byf)
```