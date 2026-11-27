The panel has hourly glucose, MAP and drug rates. The states file has one row per bundle, at hour fractions 0 (readings), 0.083 and 0.167 (drug-related bundles).

To define a dose "innovation" (the part of the dose not predicted by the readings), I need to know three things:
- what the 0.167 bundles are;
- what the dosing rule does in hours with no new reading;
- whether the rate is simply carried forward in those hours.

If it is carried forward, those hours carry no new exogenous variation. This run checks all of that on the training panel.

```python
import numpy as np, pandas as pd
fr=np.round(z['hour']%1,3)
print(pd.crosstab(fr,[z['has_insulin'],z['has_norepinephrine']]))
P=Ptr.copy()
P['Gany']=P['glucose_finger'].fillna(P['glucose_lab'])
P['Many']=P['map_arterial'].fillna(P['map_cuff'])
g=P.groupby(level=0)
for drug,rd_ in [('insulin','Gany'),('norepinephrine','Many')]:
    prev=g[drug].shift(1)
    on=P[drug].notna()&(P[drug]>0)
    nor=P[rd_].isna()
    m=on&nor
    print(drug,'on-hours',on.sum(),'no-reading frac',round(m.sum()/on.sum(),3),
          'same as prev (no reading)',round((P.loc[m,drug]==prev[m]).mean(),3),
          'same as prev (reading)',round((P.loc[on&~nor,drug]==prev[on&~nor]).mean(),3),
          'prev missing (no reading)',round(prev[m].isna().mean(),3))
```