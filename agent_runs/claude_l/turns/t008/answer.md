About 48% of stays have an arterial line, and that only partly overlaps with NE: 8,080 have both, 6,345 have an arterial line but no NE, and 5,094 have NE but no arterial line. Each reading type is present in only part of the hours (finger ~52%, lab ~20%, arterial ~46%, cuff ~61%), so when a dose is chosen the latest reading is often from an earlier hour.

Now the dosing policy itself. For insulin I'll relate the rate at hour t to the most recent glucose reading (finger or lab, carried forward), to the previous hour's rate, and to whether a fresh reading exists at hour t. The question is whether titration responds to readings in a near-deterministic way (for example, rate changes only when a new reading arrives) or has large unexplained noise. I'll do the same for NE against MAP.

```python
def add_last(P, col):
    s = P[col].groupby(level=0).ffill()
    return s
P['g_last'] = P['glucose_finger'].groupby(level=0).ffill()
P['gl_last'] = P['glucose_lab'].groupby(level=0).ffill()
P['ma_last'] = P['map_arterial'].groupby(level=0).ffill()
P['mc_last'] = P['map_cuff'].groupby(level=0).ffill()
P['ins_prev'] = P['insulin'].groupby(level=0).shift(1)
P['ne_prev'] = P['norepinephrine'].groupby(level=0).shift(1)
d = P.reset_index()
d = d[d.ih>=0]
# insulin vs latest finger glucose, by whether fresh reading
d['gbin'] = pd.cut(d.g_last, [0,100,120,140,160,180,200,220,250,300,1000])
print(d.groupby('gbin').insulin.agg(['mean','std', lambda x:(x>0).mean(),'count']))
# change in insulin when previous was >0
dd = d[(d.ins_prev>0)]
dd['dins'] = dd.insulin - dd.ins_prev
dd['fresh'] = dd.glucose_finger.notna()
print(dd.groupby(['fresh','gbin']).dins.agg(['mean','std','count']).round(3))
print(dd.dins.value_counts().sort_index().head(30))
```