Insulin follows a clear sliding scale on the glucose_finger reading taken that hour. There is none at or below 140. Above that the mean rate is about 1.05 × (1, 2, 3, 4, 6) u/h, stepping up at roughly 180, 220, 260 and 300. The spread within each step is proportional to the step (CV ≈ 0.33), so the rate looks like step(G) × a random multiplier, rounded to 0.5.

Norepinephrine is given almost always below MAP 65, sometimes between 65 and 75, and rarely above 80. When it is given at high MAP, the rate is high, which suggests titration that carries over from hour to hour rather than a pure function of the current MAP.

The random multiplier matters. If it is independent of the patient's state, it is a natural experiment for the insulin effect. If it is tied to unrecorded patient traits, it is confounded.

Next I will look at the insulin rule in finer detail. I want the exact thresholds, what happens when no finger reading was taken that hour, and the rate divided by the step value. Then I will check whether that multiplier is constant within a stay (a clinician or protocol level) or varies hour to hour.

```python
cur = P[P.gf.notna()].copy()
fb = cur.groupby(pd.cut(cur.gf,np.arange(130,200,2.5)), observed=True).agg(n=('ins0','size'), p_ins=('ins', lambda x: x.notna().mean()), mean_rate=('ins','mean'))
print(fb.to_string())
# insulin when gf not measured this hour
nm = P[P.gf.isna() & P.gf_last.notna()]
print('no gf this hour: p_ins', nm.ins.notna().mean(), 'n', len(nm))
print(nm.groupby(pd.cut(nm.gf_last,bins), observed=True).agg(n=('ins0','size'), p_ins=('ins', lambda x: x.notna().mean()), mean_rate=('ins','mean')).to_string())
# insulin when gl measured but gf not
nm2 = P[P.gf.isna() & P.gl.notna()]
print('gl only this hour: p_ins', nm2.ins.notna().mean(), len(nm2))
```