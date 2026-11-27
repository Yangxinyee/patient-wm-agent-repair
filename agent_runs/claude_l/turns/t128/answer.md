The dose-edit file gives, per stay-hour and edit, the model's mean shift and whether the drug had already run (`reached`).

I'll compute my estimated true shift for each row: the open-loop response (lags 1–12 for insulin, 1–6 for norepinephrine) applied to the recorded doses changed from hour 12. Then I'll compare model and truth averaged over reached rows, by hour, for insulin ×2.

- **If the model's ratio to truth drifts upward over the hours** (its response keeps growing while the true one levels off), a single-lag fix calibrated on the whole window will be wrong at other start hours.

```python
IRF_I=np.array([-2.06,-2.45,-2.18,-1.77,-1.29,-1.02,-0.76,-0.61,-0.42,-0.24,-0.22,-0.12])
IRF_N=np.array([3.30,2.65,1.86,0.93,0.54,0.31])
def true_shift(drug, si_, t_, fac, start=12, irf=None):
    out=np.zeros(len(t_))
    for k,g in enumerate(irf,1):
        s=t_-k; okk=(s>=start)
        dd=np.where(okk, Dtu[drug][si_, np.clip(s-H0,0,NH-1)],0.0)
        out+=g*(fac-1)*dd
    return out
de['si']=de.stay_id.map(sidx_tu)
for ed,drug,fac,irf in [('insulin_x2','insulin',2,IRF_I),('insulin_half','insulin',.5,IRF_I),('ne_x2','norepinephrine',2,IRF_N),('ne_half','norepinephrine',.5,IRF_N)]:
    m=de[(de.edit==ed)&de.reached].copy()
    m['true']=true_shift(drug,m.si.values,m.hour.values,fac,irf=irf)
    print(ed,'n',len(m),'model %.2f true %.2f'%(m['shift'].mean(),m['true'].mean()))
    if ed=='insulin_x2':
        g=m.groupby('hour')[['shift','true']].mean(); g['ratio']=g['shift']/g['true']
        print(g.round(2).T.to_string())
```