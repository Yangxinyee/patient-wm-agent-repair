Run 196 showed that a single linear readout on the FFN input is not good enough:
- **Gates are not separable:** insulin presence has 7 + 132 misclassified rows. Norepinephrine is much worse (226 + 1311).
- **Insulin readout is noisy:** its error sd (0.60) is as large as the true dose surprise (0.63).
- **Readouts are not linear in the dose:** at insulin x3 the bias is −1.08 and the rms 2.2, and at norepinephrine x0 the bias is +0.48.
- **Compensation leaves a band bias:** at m=1 the residual by glucose band is −0.08, +0.13, +0.56, +0.99, as expected when a linear fit approximates the policy's step function.

The cause is that the dose information in norm3(x1) is a small, state-entangled part of a large residual stream. That is why the earlier contexts carried the dose through head 1's value path, which is exact and does not depend on the state.

The remaining open question is how much state-dependent compensation the factual-fit limit actually requires. Before building anything complex I will estimate it offline from the records. I need the readings with the original model's predictions (fwd_2) and the doses of the same tuning stays. For each candidate injection k·(rate − c) I will apply the model's factual persistence of 0.403 per hour and compute the change in Gaussian log loss from the observed residuals. The candidates for c are a constant, the band mean, and a linear approximation.

This run only checks what data is available: the column layouts of fa, drugs_tu and tuning.csv, and the residual sd per reading type.

```python
import pandas as pd
print("fa:", type(fa).__name__, getattr(fa, 'columns', None).tolist() if hasattr(fa, 'columns') else '', getattr(fa, 'shape', ''))
print(fa.head(3).to_string() if hasattr(fa, 'head') else '')
print("drugs_tu:", type(drugs_tu).__name__, drugs_tu.columns.tolist() if hasattr(drugs_tu, 'columns') else '')
print(drugs_tu.head(3).to_string() if hasattr(drugs_tu, 'head') else '')
tu = pd.read_csv('/data/tuning.csv', nrows=5); print("tuning.csv cols:", tu.columns.tolist()); print(tu.head(5).to_string())
rd_ = open('/data/README.md').read(); print(rd_[:1800])
if hasattr(fa, 'columns') and 'observed' in fa.columns:
    for evn_, g_ in fa.groupby('event'):
        r_ = g_['observed'] - g_['predicted_mean']; print(evn_, "n %d resid mean %+.2f sd %.2f  nll %.3f" % (len(g_), r_.mean(), r_.std(), g_[[c for c in g_.columns if 'nll' in c.lower()][0]].mean() if any('nll' in c.lower() for c in g_.columns) else np.nan))
print("ids100 sample", ids100[:3], " sel100 file:", open('/work/sel100_both.json').read()[:200])
```