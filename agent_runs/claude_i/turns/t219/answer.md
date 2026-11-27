My encoder reproduces the tool's drug encodings exactly. The randomized-dose dataset is built: 10,244 readings from 100 stays × 4 random edits, each with the head's normalised input and the insulin and norepinephrine deviations at lags 1–4.

Now I fit ridge readouts with cross-validation that holds out whole stays. For each lag and drug I report held-out R² on three sets:
- all readings;
- readings where the target is just the factual deviation (unedited);
- readings where the edit changed it.

The aim is to see how well the state carries the dose history and how noisy the readout would be on factual records, where its errors cost NLL.

The plan holds if:
- the edited-subset R² is high (≥0.7) at lags 1–2;
- the factual-subset error is small compared with the deviation spread (insulin about 0.74, norepinephrine about 0.17).

If the edited R² is low, the state does not carry the dose history well enough for this readout.

```python
from sklearn.linear_model import Ridge
grp_s = MET[:,0]; ust = np.unique(grp_s); folds = np.array_split(np.random.RandomState(0).permutation(ust), 5)
def cvpred(Xm, y, lam):
    pr = np.zeros_like(y)
    for f_ in folds:
        te = np.isin(grp_s, f_); m_ = Ridge(alpha=lam).fit(Xm[~te], y[~te]); pr[te] = m_.predict(Xm[te])
    return pr
def r2(y, p): return 1 - ((y-p)**2).mean()/max(y.var(),1e-12)
s0v, fIv, fNv, tv = MET[:,2], MET[:,3], MET[:,4], MET[:,5]
names = [f'{d}{k}' for k in range(1,5) for d in ('I','N')]
for lam in [1.0, 10.0]:
    out_ = []
    for c in range(8):
        y = TT[:,c]; p = cvpred(QQ, y, lam); kk = c//2+1; fv = fIv if c%2==0 else fNv
        fact = (tv-kk < s0v) | (fv==1); ed = ~fact
        sl = np.polyfit(p[ed], y[ed], 1)[0] if ed.sum()>10 else np.nan
        out_.append(f"{names[c]}: all {r2(y,p):.2f} fact {r2(y[fact],p[fact]):.2f} (err sd {np.std(y[fact]-p[fact]):.2f}, y sd {np.std(y[fact]):.2f}) ed {r2(y[ed],p[ed]):.2f} slope {sl:.2f}")
    print('lam', lam); print('\n'.join(out_))
```