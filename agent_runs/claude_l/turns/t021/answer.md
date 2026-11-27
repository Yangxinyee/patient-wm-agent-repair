The original model's answers are small: insulin x2 gives −3.2 mg/dL, x1/2 +1.8 mg/dL, NE x2 +0.6 mmHg and NE x1/2 −0.8 mmHg.

For insulin these are far below my rough record-based expectation for x2, which was a 10–25 mg/dL drop. That suggests the model's causal response to insulin is too weak, perhaps partly cancelled by the dose being read as evidence of high glucose.

The readout pools 669 "reached" stay-hours over 75 insulin stays (about 9 per stay) and 549 over 50 NE stays. Next I'll inspect the per-stay-hour CSV to pin down exactly which rows count, whether the mean is pooled over rows or averaged per stay, and what other columns exist. Then I can compute my own estimate of the true effect on exactly the same rows. I'll also save the first findings to the notebook.

```python
import glob
print(glob.glob('/work/fwd_1/*'))
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.shape); print(de.columns.tolist()); print(de.head(30).to_string())
print(de.describe().T)
open('/work/notes.md','a').write("""
## Data findings (runs 4-18)
- All stays 25h. Readings integer hours, infusions start t+5min, 1h. Insulin rate multiples of 0.5 (dose=rate); NE rate 0.01 steps capped at 1.0 (dose=4.8*rate).
- Insulin policy: ~0 below 140, partial 140-160, always >160; GBM on recorded history R2=0.82, resid sd 0.49 (multiplicative), resid iid (autocorr 0.004), no stay effect.
  Negative control: resid does NOT predict same-hour lab glucose given finger -> no hidden-info confounding in teacher forcing.
- NE policy R2=0.93 resid sd .09 (.137 on NE); negative control fine.
- One-hour effects via iid dose residual: insulin -2.06 mg/dL per U/h on next finger glucose; NE +3.30 mmHg per mcg/kg/min on next art MAP.
- Insulin open-loop impulse response (local projection + feedback removal), per 1 U/h for 1h at lag k=1..12:
  -2.06 -2.45 -2.18 -1.77 -1.29 -1.02 -0.76 -0.61 -0.42 -0.24 -0.22 -0.12  (sum ~ -13). Transient: glucose mean-reverts to a baseline insulin doesn't move.
- Insulin effect scales ~ proportional to glucose level (about -0.0097*G per U/h, 3h avg). Looks like minimal model X*G.
- Finger vs lab: no bias, diff sd 9 (each noise ~6). Cuff = arterial + 3, diff sd 6.7.
## Forward 1: dose_edit original, tuning batch1 (100 stays in /work/batches.json)
- ins x2 -3.25 (669 stay-hrs, 75 stays), ins x1/2 +1.77, NE x2 +0.62 (549, 50 stays), NE x1/2 -0.82.
""")
```