Forward pass 1 gives the original model's answers on these stays:
- insulin ×2: −1.88 mg/dL (50 stays, 456 stay-hours);
- insulin ×½: +0.98;
- norepinephrine ×2: −0.41 mmHg (27 stays);
- norepinephrine ×½: −0.52.

The insulin answers have the right sign but look several times too small against my data-based expectation. The norepinephrine answers are wrong: ×2 lowers MAP and both edits have the same sign, consistent with the rate acting as a proxy for sickness rather than a cause.

Next I'll look at the per-stay-hour CSV to learn exactly which stay-hours count as "reached" and how the shift is laid out. Then I can compute my data-based truth on exactly the same footing. I'll also record these results in the notebook.

```python
import glob
f1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(f1.shape); print(f1.columns.tolist()); print(f1.head(20).to_string())
print(f1.describe(include='all').T.to_string()[:3000])
with open('/work/notes.md','a') as fh:
    fh.write("""
## Findings from records (runs 2-17)
- All stays 25.17h; readings on the hour, infusions at h+0.083 (1h). Weight=80kg for all (NE dose=rate*4.8mg).
- INSULIN policy: given iff glucose_finger(h)>=150; base rate tier 1/2/3/4 for G in [150,180),[180,220),[220,250),>=250; rate=base*mult rounded to 0.5, mult iid (lag1 corr 0.006, uncorrelated w/ trend) => within-tier rate is randomized.
- Insulin causal effect (within-tier OLS, then deconvolved for feedback on later doses): psi per unit/h at lags1..8 = -2.0,-2.3,-2.0,-1.6,-1.1,-0.9,-0.6,-0.4 (sum ~-11; lab ~-12). ~linear in rate.
- NOREPI policy: start at ~0.05 when MAP<65; titrate multiplicatively: MAP<65 up, 65-75 hold, >=75 down; cap 1.0; stop from low rates when MAP>=75. noise ~exogenous.
- NE causal effect on map_arterial (ctrl N_prev, MAP, zone; deconvolved for strong persistence C1=0.75): psi per mcg/kg/min-hour lags1..6 = 3.4,2.6,1.75,0.9,0.4,0.2 (sum ~9.1).
- Confounding signal: given N(t), MAP(t), higher N_prev predicts LOWER MAP (proxy of severity).
## Forward 1 (dose_edit, original, tuning sel1 in /work/sel1.json): ins_x2 -1.88, ins_half +0.98, ne_x2 -0.41, ne_half -0.52 (NE wrong sign/both negative).
""")
```