Norepinephrine shows the same pattern as insulin.
- **Open-loop response:** per 1 mcg/kg/min for one hour it is +3.30, +2.65, +1.86, +0.93, +0.54, +0.31 mmHg, then about 0. The effect is shorter-lived than insulin's (timescale about 1.5–2 h) and integrates to about +9.8 mmHg·h.
- **Feedback:** unlike insulin, NE dosing titrates from the previous rate. A residual at t carries into the next dose with weight 0.73, so the closed-loop response looks much longer than the drug's own effect.
- **Batch 1 estimate:** NE x2 is about +4.9 mmHg and x1/2 about −2.5 mmHg, against the model's +0.62 and −0.82.

So the model is several-fold too weak for both drugs. One caveat: NE x2 pushes rates above the 1.0 cap seen in the records, so the truth there is an extrapolation.

Next I'll compare, row by row, my estimated true shift with the model's shift as a function of hours since hour 12 and of dose. That shows whether the model is uniformly too weak or has the wrong time profile (for example too slow an onset, or accumulating like a random walk). I'll also record the findings in the notebook.

```python
def est_rows(de, edit, drug, hk, mult, scale_col=None):
    rr = de[(de.edit==edit)&de.reached][['stay_id','hour','shift','factual']].copy()
    rr['est'] = [true_shift_lin(PT.loc[s], mult, drug=drug, hk=hk, scale_col=scale_col)[hh] for (s,hh) in rr[['stay_id','hour']].values]
    return rr
ri2 = est_rows(de, 'insulin_x2', 'insulin', hk, 2.0)
rn2 = est_rows(de, 'ne_x2', 'norepinephrine', hnk, 2.0)
print("insulin x2 by hour: model vs est"); print(ri2.groupby('hour')[['shift','est']].mean().round(2).T)
print("NE x2 by hour"); print(rn2.groupby('hour')[['shift','est']].mean().round(2).T)
print("corr per row ins:", np.corrcoef(ri2['shift'], ri2.est)[0,1], " slope:", np.polyfit(ri2.est, ri2['shift'],1))
print("corr per row NE:", np.corrcoef(rn2['shift'], rn2.est)[0,1], " slope:", np.polyfit(rn2.est, rn2['shift'],1))
open('/work/notes.md','a').write("""
- Readout def: rows only at hours 13-24 with a RECORDED reading of the outcome code; reached = a drug record started >=12 and before that hour; mean pooled over reached rows.
- My 'truth' (linear open-loop IRF applied to recorded doses): batch1 ins x2 ~ -14.2 (G-scaled -13.2), x1/2 ~ +7; model -3.25/+1.77 -> model ~4x too weak.
- NE open-loop IRF per (mcg/kg/min * 1h): 3.30 2.65 1.86 .93 .54 .31 ~0 (sum 9.8). NE policy titrates from prev rate (D1=.73).
  batch1 NE x2 est +4.9, x1/2 -2.5; model +0.62/-0.82. Caveat: NE x2 exceeds observed max rate 1.0 (cap).
""")
```