Generating samples was fast: 1,500 training stays in 69 s, giving 39,178 glucose_finger and map_arterial prediction points with dose-perturbed states. All of them are saved to /work.

Now the key test: can the head input carry the actual recent doses well enough to support a head-level edit? I fit ridge decoders from the 192-dimensional head input to:
- for finger glucose, the insulin rates started 1, 2 and 3 hours earlier;
- for arterial MAP, the corresponding NE rates.

I use cross-validation grouped by stay, and report R² and the calibration slope separately for perturbed and unperturbed stays.

What would make the head-level design viable:
- R² well above 0.8 for the lag-1 dose, in both the perturbed and the unperturbed stays;
- a slope near 1, meaning the decoder truly tracks the dose rather than the glucose-policy link;
- binned decoder means that are close to linear across the ×½ to ×2 range.

Low R² or strong curvature would argue for an upstream channel instead.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
pidx = {s:i for i,s in enumerate(train_ids[:1500])}
def fit_dec(code, lagcol, alpha=1.0):
    S_ = [x for x in SAMP if x[0]==code]
    Xw = np.array([x[8] for x in S_]); y = np.array([x[lagcol] for x in S_]); g = np.array([x[1] for x in S_])
    pert = np.array([pidx[x[1]] % 4 != 0 for x in S_])
    pred = np.zeros_like(y)
    for trn, tst in GroupKFold(5).split(Xw, y, g):
        m = Ridge(alpha=alpha).fit(Xw[trn], y[trn]); pred[tst] = m.predict(Xw[tst])
    def r2(a,b): return 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    sl_p = np.polyfit(y[pert], pred[pert], 1)[0]; sl_u = np.polyfit(y[~pert], pred[~pert], 1)[0]
    print(code, 'lag', lagcol-4, 'n', len(y), 'R2 pert %.3f unpert %.3f | slope pert %.3f unpert %.3f | resid sd pert %.3f unpert %.3f' %
          (r2(y[pert],pred[pert]), r2(y[~pert],pred[~pert]), sl_p, sl_u, (y-pred)[pert].std(), (y-pred)[~pert].std()))
    return y, pred, pert, Xw, g
for code in ['glucose_finger','map_arterial']:
    for lc in [5,6,7]:
        out_ = fit_dec(code, lc)
        if lc==5:
            y, pred, pert, _, _ = out_
            bins_ = np.quantile(y[y>0], [0,.2,.4,.6,.8,1.0])
            print('   binned (y>0, pert): ', [(round(y[pert&(y>=a)&(y<=b)].mean(),3), round(pred[pert&(y>=a)&(y<=b)].mean(),3)) for a,b in zip(bins_[:-1],bins_[1:])], ' y=0 mean pred %.3f' % pred[y==0].mean())
```