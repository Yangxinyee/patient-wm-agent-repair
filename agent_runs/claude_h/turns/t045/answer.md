The paired factual pass shows the dose-invariant edit costs essentially nothing:

| | Original | Edited | Change |
|---|---|---|---|
| Value NLL per reading (overall) | 1.34364 | 1.34369 | +0.00005 |
| glucose_finger NLL | 1.2873 | 1.2874 | +0.0001 |
| glucose_lab NLL | 1.0394 | 1.0389 | −0.0005 |
| map_arterial NLL | 1.2612 | 1.2614 | +0.0002 |
| map_cuff NLL | 1.6466 | 1.6467 | +0.0001 |

The overall change is about 1% of the 0.005 allowance. Mean absolute errors are unchanged to the third decimal. So the model did not rely on dose magnitude to predict readings, which fits doses being functions of already-recorded readings plus random noise.

The edit is therefore safe. What remains is whether the true effects are really zero. Insulin looks solidly absent. For norepinephrine I have a weak, inconsistent negative signal: an implied ×2 effect of about −0.32 ± 0.14, but +0.05 ± 0.11 in hours ≥ 12, the period that matters.

Before deciding, I will record this result in the notebook and look at the panel structure. That sets up a sharper, variance-reduced norepinephrine analysis using richer covariates on all 30,000 stays.

- **Supports leaving norepinephrine at zero:** that analysis shrinks the norepinephrine estimate toward zero with a smaller standard error.
- **Supports installing a small effect:** it confirms a stable negative effect, especially after hour 12.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 39-43 (after restart 1)
- Current edit (Plan B, dose-invariant): embed.bins.dose & embed.bins.rate rows 1..16 := original row 8 (median bin); row 0 untouched.
  embed.numeric.weight cols z_dose (2) and z_rate (4) zeroed. Check: drug-atom embedding term identical for x1, x2, x1/2 (dev 1.5e-8).
- FWD 2 factual both, tuning sel1 (100 stays, 5323 values): original 1.34364, edited 1.34369 (+0.00005). By event all within +-0.0005. => NLL SAFE.
- Next: sharpen NE evidence (variance-reduced LP with richer covariates), then dose_edit both to confirm edited readouts = 0.
""")
import pickle, pandas as pd
P=pd.read_pickle('/work/panel_train.pkl')
print(type(P))
if isinstance(P,dict):
    for k,v in P.items(): print(k, type(v), getattr(v,'shape',None))
else:
    print(P.shape); print(P.columns.tolist()[:80]); print(P.head(3).T)
print([k for k,v in globals().items() if isinstance(v,(pd.DataFrame,np.ndarray)) and not k.startswith('_')][:60])
```