The saved candidate (a_ins = −1.9, a_ne = +3) gets the head-tensor names right and round-trips exactly. But its offline factual cost is 0.018 nats per reading, more than three times the 0.005 budget, so I will not test it as it stands. The cost comes mostly from readout noise:

| Rows | Extra NLL per reading |
|---|---|
| Cuff MAP | 0.015 to 0.06 |
| Arterial MAP | 0.005 to 0.019 |
| Finger glucose | 0.010 to 0.020 |
| Lab glucose | about 0 |

My hypothesis is that much of this noise lands on readings where no drug was given in the preceding bundle. There the true correction is zero, so any shift is pure noise. If so, a second gate in the units ("the last bundle held this drug") could cut the cost a lot without changing the counterfactual response. Drug records fall at the same times in both rollout arms, so the gate is identical in both.

The check: split the cost and the squared shift by code and by whether the preceding bundle held insulin or norepinephrine. If the drug-free rows carry most of the cost, gating is worth building. If the cost sits in the drug rows, I need better readouts instead.

```python
import numpy as np
E=make_edit(-1.9,3.0)
rows_all=[]
for i in range(5):
    Xs_,R_=datasets[i]; z=np.load(files[i]); hi_=z['has_insulin'].astype(bool); hn_=z['has_norepinephrine'].astype(bool)
    prev_=R_['prev'].values.astype(int)
    o1=head_run(Xs_,R_,E1); oE=head_run(Xs_,R_,E); pmb,nllb,_,_=base_out[i]
    df_=pd.DataFrame({'code':R_['code'].values,'hi':hi_[prev_],'hn':hn_[prev_],'sh1':npv(o1[0])-pmb,'dnE':npv(oE[1])-nllb,
                      'dev1':R_['dev1'].values,'nedev1':R_['nedev1'].values})
    rows_all.append(df_)
A_=pd.concat(rows_all,ignore_index=True)
A_['drug']=np.where(A_.code.isin([6,10]),A_.hi,A_.hn)
g_=A_.groupby(['code','drug']).agg(n=('dnE','size'),dNLL=('dnE','mean'),sh1_sq=('sh1',lambda s:(s**2).mean()),
                                   dev1_sd=('dev1','std'),nedev1_sd=('nedev1','std'))
print(g_.round(4))
tot=len(A_)
print('share of total cost by (code,drug):'); print((A_.groupby(['code','drug']).dnE.sum()/A_.dnE.sum()).round(3))
print('overall dNLL',round(A_.dnE.mean(),4),' if off-rows had zero cost:',round(A_.dnE.where(A_.drug,0).mean(),4))
```