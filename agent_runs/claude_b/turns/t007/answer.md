The panel is clean. Readings run from hour −1 to 24, with glucose_finger present about 55–62% of the time, glucose_lab 20%, map_arterial 46% and map_cuff 61%. Drugs start at hours 0–24: insulin in 34–46% of stay-hours, rising through the stay, and norepinephrine in about 33%. Many hours have no glucose reading, yet insulin still continues through them. So clinicians dose without a fresh recorded reading. If the dose then tracks something not in the record, such as the true glucose, that is exactly the hidden confounding I need to worry about.

Next I will characterise the dosing policy: how the presence and rate of insulin depend on the current glucose_finger reading and the previous rate, and likewise norepinephrine on MAP. I also want to know whether stays split into arterial-line and cuff-only groups. If the dose varies a lot at a fixed reading and previous rate, there is random dose variation to identify effects with. If it is almost deterministic, I will need other strategies.

```python
P = P.loc[(slice(None), slice(-1,24)), :].copy()
P['ins'] = P['insulin'].fillna(0.0); P['ne'] = P['norepinephrine'].fillna(0.0)
g = P.groupby(level=0)
P['ins_prev'] = g['ins'].shift(1); P['ne_prev'] = g['ne'].shift(1)
# stay-level arterial line
art = P.map_arterial.notna().groupby(level=0).mean()
print("per-stay frac hours with map_arterial:", np.histogram(art, bins=[0,0.01,0.2,0.5,0.8,0.99,1.01])[0])
cuf = P.map_cuff.notna().groupby(level=0).mean()
print("per-stay frac hours with map_cuff:", np.histogram(cuf, bins=[0,0.01,0.2,0.5,0.8,0.99,1.01])[0])
gf = P.glucose_finger.notna().groupby(level=0).mean()
print("per-stay frac hours glucose_finger:", np.histogram(gf, bins=[0,0.01,0.2,0.5,0.8,0.99,1.01])[0])
# insulin policy vs current glucose_finger
Q = P[P.glucose_finger.notna()].copy()
Q['gbin'] = pd.cut(Q.glucose_finger, [0,100,120,140,160,180,200,220,250,300,400,1000])
print(Q.groupby('gbin', observed=True).agg(n=('ins','size'), p_ins=('ins', lambda x:(x>0).mean()), mean_rate=('ins', lambda x: x[x>0].mean())).round(3))
print(Q.groupby(['gbin', pd.cut(Q.ins_prev,[-0.1,0.1,1,2,3,5,100])], observed=True).ins.agg(['size','mean','std']).round(2).unstack().to_string())
```