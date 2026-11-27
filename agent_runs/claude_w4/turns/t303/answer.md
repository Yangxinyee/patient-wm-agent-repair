The v4 edit badly damages factual fit, so it fails outright as built. Value NLL rose from 1.4938 to 1.6100 (+0.116 nats); the allowed margin is +0.005.

| Reading | Original NLL | v4 NLL | Change | Mean abs error, original → v4 |
|---|---|---|---|---|
| glucose_finger | 1.5515 | 1.6887 | +0.137 | 11.3 → 11.8 |
| glucose_lab | 1.2396 | 1.4802 | +0.241 | 9.1 → 10.2 |
| map_arterial | 1.3522 | 1.3987 | +0.047 | 3.9 → 4.2 |
| map_cuff | 1.6918 | 1.8145 | +0.123 | 6.1 → 6.6 |

Mean absolute errors rose only modestly. So the edit mainly distorts the shape of the predictive distributions, probably their spread, rather than just biasing the means. That also fits the rollout drift: a widened, right-skewed glucose distribution sampled 20 times pushes the mean up. MAP readings got worse too, so the damage is not confined to doses.

Before rebuilding, I need to know where the damage enters:
- If it is just as large in stays that never receive any drug, it enters through reading bundles. The `b_in` channel being non-quiet there is the prime suspect, and channel directions quiet at both bundle types would fix it.
- If it appears only after drug bundles, the injected state change itself is not a clean "glucose level" change. Then the target direction must be redesigned, for example using the level directions instead of the news directions, as the unfinished v5 parts intended.

The next run merges the two factual files and compares the change in NLL and in predicted mean across three groups of readings: stays with no drugs, readings before the first drug, and readings after drugs. For finger glucose it also regresses the change in predicted mean on the preceding hour's centred insulin dose.

```python
import pandas as pd, numpy as np
fo = pd.read_csv('/work/fwd_9/factual_original.csv'); fe = pd.read_csv('/work/fwd_9/factual_edited.csv')
print(fo.columns.tolist(), fo.shape)
key = [c for c in fo.columns if c in ('stay_id','hour','event','value','time','t')]
mcol = [c for c in fo.columns if 'mean' in c][0]; ncol = [c for c in fo.columns if 'nll' in c][0]
f9 = fo.merge(fe, on=key, suffixes=('_o','_e')); f9['dm'] = f9[mcol+'_e']-f9[mcol+'_o']; f9['dn'] = f9[ncol+'_e']-f9[ncol+'_o']
drugs = tun[tun.event.isin(['insulin','norepinephrine'])]
first_drug = drugs.groupby('stay_id').hour.min()
f9['fd'] = f9.stay_id.map(first_drug).fillna(99.)
f9['grp'] = np.where(f9.fd>90, 'no_drug_stay', np.where(f9.hour < f9.fd, 'before_drug', 'after_drug'))
print(f9.groupby(['event','grp']).agg(n=('dn','size'), dNLL=('dn','mean'), dmean=('dm','mean'), absdm=('dm', lambda x: np.abs(x).mean())).round(3).to_string())
# finger: regress dm on previous-hour insulin (centred) and NE, within after_drug readings
fg = f9[(f9.event=='glucose_finger')].copy(); fg['h'] = np.floor(fg.hour+1e-6).astype(int)
def rate_at(R, s, h): return R.loc[s, h] if (s in R.index and 0 <= h < 26) else 0.
fg['rI1'] = [rate_at(RI, s, h-1) for s, h in zip(fg.stay_id, fg.h)]; fg['rN1'] = [rate_at(RN, s, h-1) for s, h in zip(fg.stay_id, fg.h)]
fg['cI'] = fg.rI1 - 1.7705*(fg.rI1>0)
X = np.c_[np.ones(len(fg)), fg.cI, (fg.rI1>0), fg.rN1>0]
b = np.linalg.lstsq(X, fg.dm.values, rcond=None)[0]; print('finger dm ~ 1 + cI + presI + presN:', b.round(3))
b = np.linalg.lstsq(X, fg.dn.values, rcond=None)[0]; print('finger dNLL ~ 1 + cI + presI + presN:', b.round(4))
print('dNLL quantiles (all):', np.percentile(f9.dn, [5,25,50,75,95]).round(3), ' corr(dn, |dm|)', np.corrcoef(f9.dn, np.abs(f9.dm))[0,1].round(2))
```