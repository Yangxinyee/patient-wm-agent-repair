## Step 14: the threshold design confirms about −2 mg/dL per unit/h after one hour

**What the last result showed.** The scaled jumps at each threshold give the one-hour insulin effect:

| threshold | effect per unit/h (SE) | note |
|---|---|---|
| 180 | −1.76 (0.18) | |
| 220 | −2.0 to −2.1 (0.3) | |
| 150 | −2.1 to −2.4 (0.3) | the chance of a next-hour reading also jumps by 0.68 here, because hourly checks start once insulin is on |
| 260 | −3.2 to −4.1 (1–1.5) | imprecise |

These agree with the within-band-noise regression (−2.0). So the per-record dose noise behaves as exogenous, and insulin truly lowers glucose: about 2 mg/dL per unit/h after one hour.

**What I will do next.** Record this in the notebook. Then estimate the full dynamic response, meaning how a one-hour dose affects glucose 1, 2, 3 … hours later, holding later doses fixed:
- Use the dose "surprises" (dose minus its band mean) at hours t, t+1, … as instruments for the doses in a just-identified two-stage least squares regression.
- This handles the policy feedback.

I will also look at whether the response is linear in dose.

**What would count as evidence.** Positive persistence (lagged effects that do not vanish) would mean the effect of sustained dosing accumulates over hours 13–24, and the true readout could be far larger than −2 × dose.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 2-14: data facts
- All stays 25.17h; readings at integer hours, drugs at h+0.0833 (1h infusions). 30k stays; ins in 18.4k stays, NE in 13.1k.
- INSULIN POLICY (sliding scale on latest glucose reading; finger preferred if same hour as lab): rate=0 if G<=150,
  base 1 (150-180], 2 (180-220], 3 (220-260], 4 (>260), times iid lognormal noise (log-sd ~0.31-0.37) rounded to 0.5, min 0.5.
  Noise has ~zero autocorrelation (per-record). Finger checks become hourly when G>150.
- finger vs lab same hour: mean diff 0.03, sd 9.1 -> same latent glucose. arterial ~3 mmHg below cuff (sd 6.7).
- NE: rate 0.01..1.0 (cap at 1.0: 14% of NE records), dose=4.8*rate. NE on-prob falls smoothly with MAP; sharper near art MAP 64-65.
- INSULIN 1-HOUR EFFECT: within-band-noise OLS: -2.0 mg/dL per unit/h on next-hour finger glucose (se .05), stable across controls.
  RD at thresholds 150/180/220/260: -2.1..-2.4, -1.76, -2.0, -3.2..-4 (se .2-1.0) -> consistent, ~ -2 per unit/h at 1h.
""")
# Build dose surprise: e = rate - E[rate | band]
x = x.sort_values(['stay_id','h'])
band_mean = x[x.ins_rate>0].groupby('band').ins_rate.mean()
print(band_mean)
x['ins_exp'] = np.where(x.band>0, x.band.map(band_mean).fillna(0), 0.0)
# careful: where latest glucose is NaN (no reading yet), band=0 -> expected 0
x['e_ins'] = x.ins_rate - x.ins_exp
print("e_ins stats when on:", x.loc[x.ins_rate>0,'e_ins'].describe().round(3).to_dict())
print("frac rate>0 with band 0:", ((x.ins_rate>0)&(x.band==0)).mean())
```