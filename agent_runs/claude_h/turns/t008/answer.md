The insulin policy is a sliding scale with multiplicative noise. The scale has base levels of
- about 0 below 150,
- 1 for 150–180,
- 2 for 180–220,
- 3 for 220–260,
- 4 for 260–300,
- 6 above 300,

and the given rate spreads widely around each base. For example, in 150–180 the rate is 0.5, 1, 1.5 or 2 with probabilities of roughly 0.17, 0.6, 0.2 and 0.03, and about 3% of eligible hours get nothing.

This is promising for identification. The jitter within a bin, and the jumps at the thresholds (a regression-discontinuity design), are both sources of dose variation that may be unrelated to the patient's hidden state. The example stay suggests that when no glucose reading is taken in an hour, no insulin is given, and that glucose_lab readings also drive the scale.

Next I will verify what the policy reads:
- the finger reading versus the lab reading;
- what happens when no glucose reading exists that hour;
- any dependence on the previous rate.

I will do the same for norepinephrine against MAP (arterial versus cuff). This tells me what to condition on.

```python
P['glu_any'] = P.gf.combine_first(P.gl)
P['has_gf']=P.gf.notna(); P['has_gl']=P.gl.notna()
def scale(gv):
    return np.select([gv<150, gv<180, gv<220, gv<260, gv<300], [0,1,2,3,4], 6)
d = P.copy()
d['case'] = np.where(d.has_gf & d.has_gl, 'both', np.where(d.has_gf,'finger', np.where(d.has_gl,'lab','none')))
print(d.groupby('case').ins.agg(['mean', lambda x:(x>0).mean(), 'size']))
# when lab only: does scale on lab predict?
dl = d[d.case=='lab']
print(pd.crosstab(pd.cut(dl.gl,[0,130,140,150,160,180,220,260,300,500]), dl.ins.clip(upper=5)).to_string())
db = d[d.case=='both']
db['sf']=scale(db.gf); db['sl']=scale(db.gl)
print("both: corr ins with finger-scale vs lab-scale")
print(db.groupby(['sf','sl']).ins.agg(['mean','size']).unstack().round(2).to_string())
# none case: depends on previous rate?
dn = d[d.case=='none']
print(dn.groupby(dn.ins_prev.clip(upper=4)).ins.agg(['mean', lambda x:(x>0).mean(),'size']))
```