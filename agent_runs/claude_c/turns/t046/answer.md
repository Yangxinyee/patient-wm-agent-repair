The z posterior is almost one-dimensional. At insulin bundles a single dimension has variance 2.28 and the others are about zero. That dimension tracks the raw rate (R² 0.78) and the tier's expected rate (R² 0.78) better than the innovation (R² 0.57). So z most likely encodes the raw dose, and amplifying it would double count.

Before ruling it out, I want three things:
- the coefficients of the active dimension on innovation and on expected rate, which separate raw from innovation coding;
- how much that dimension varies at reading and norepinephrine bundles, which tells me whether an edit along it would be selective;
- the size of the factual-NLL risk of a raw-dose edit. For this I estimate the glucose prediction error sd and the share of readings that come right after insulin, using the factual file from pass 2.

Equal coefficients mean raw coding, which rules out this route. An expected-rate coefficient near zero would revive it.

```python
zi_var = X3[:, 960:992].var(0)
jd = int(np.argmax(Zi.var(0))); print("active z dim", jd, "overall var", np.round(np.sort(zi_var)[::-1][:4],3))
zz = X3[:, 960+jd]
kinds = np.where(hi3, 'ins', np.where(hn3, 'ne_only', np.where(np.round(hr3-np.floor(hr3),4)==0, 'read', 'other')))
for k_ in ['ins','ne_only','read','other']:
    m_ = kinds==k_; print(k_, m_.sum(), "mean", zz[m_].mean().round(3), "sd", zz[m_].std().round(3))
A = np.column_stack([np.ones(ok.sum()), innov[ok], exp_rate[ok], (gsrc_b[ok]-200)/50, hn3[ins_b][ok].astype(float)])
c_, *_ = np.linalg.lstsq(A, zz[ins_b][ok], rcond=None)
print("active z ~ [1, innov, exp_rate, g, has_ne]:", np.round(c_,3), "R2", round(1-np.var(zz[ins_b][ok]-A@c_)/np.var(zz[ins_b][ok]),3))
# factual file
fa2 = pd.read_csv('/work/fwd_2/factual_original.csv'); print(fa2.columns.tolist()); print(fa2.head(3))
print(fa2.groupby(fa2.columns[2] if 'code' not in fa2.columns else 'code').size())
```