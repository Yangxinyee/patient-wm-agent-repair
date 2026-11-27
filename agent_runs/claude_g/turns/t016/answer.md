Norepinephrine also shows almost no effect on map_arterial. The DML estimate is −0.33 ± 0.08 mmHg per mcg/kg/min at 1 hour (−0.37 at 2 h, −0.28 at 3 h, −0.15 ± 0.14 at 6 h). On the dose-edit scale, doubling a typical 0.5 rate would change MAP by about −0.2 mmHg, which is negligible. The small negative sign could be slight residual confounding from the titration rule. So far both drugs look inert in this ICU, at least at the margin: varying an existing dose up or down does nothing.

That leaves an alternative to rule out: a saturating (all-or-none) effect, where any insulin lowers glucose but the amount does not matter. It would leave x2/x1/2 truths at about 0 but would change my picture of the drugs. I can test it with the sharp threshold at 150 mg/dL, a regression discontinuity: readings of 151–153 always get insulin and 147–149 never do. A jump in later glucose change across the threshold would reveal an effect of being on insulin at all. No jump would confirm insulin is inert.

I will also extend the dose-margin check to lags 7–12 to cover the readout's full horizon.

```python
# Regression discontinuity at 150 for insulin presence
G['h_'] = G.index.get_level_values('h')
rd = G[G.glucose_finger.between(135,165) & (G.glucose_lab.isna() | True)].copy()
for k in [1,2,3,6]:
    rd['dG'] = rd[f'gf_p{k}'] - rd.glucose_finger
    t = rd.groupby('glucose_finger').agg(n=('dG','count'), dG=('dG','mean'), pins=('ins', lambda x: (x>0).mean()))
    t = t[t.index!=150]
    lo = t[t.index<150]; hi = t[t.index>150]
    # local linear fit each side
    bl = np.polyfit(lo.index, lo.dG, 1, w=np.sqrt(lo.n)); bh = np.polyfit(hi.index, hi.dG, 1, w=np.sqrt(hi.n))
    print(f'k={k}: left limit {np.polyval(bl,150):.2f}, right limit {np.polyval(bh,150):.2f}, jump {np.polyval(bh,150)-np.polyval(bl,150):.2f}')
print(t.round(2).T.to_string())
```