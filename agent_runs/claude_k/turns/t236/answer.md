Run 219 shows the cause. Drug bundles sit at hour t + 0.083 (5 minutes after each hour's reading bundle), while readings sit on whole hours. So the next reading comes 0.917 hours after the dose bundle, and my condition `lag >= 1` dropped every pair.

This also settles the ordering: the dose bundle follows the same-hour reading, so the dosing decision can see that reading through ctx.

I'll redo the factual-error regression with lag bins defined by ceil(continuous lag), for lags above 0. What would confirm the innovation-write plan is unchanged: clear negative insulin coefficients and positive norepinephrine coefficients at early lags, with a measurable reduction in error variance.

```python
def lagX(evn,which):
    idx_=np.where(R.event.values==evn)[0]; Xl=np.zeros((len(idx_),NL))
    for n_,i in enumerate(idx_):
        for (hb,a_,b_) in bys.get(R.stay_id.values[i],[]):
            lg_=R.hour.values[i]-hb
            if 0<lg_<=NL: Xl[n_,int(np.ceil(lg_))-1]+= (a_ if which=='A' else b_)
    return idx_,Xl
KAP={}
for evn,wh in [('glucose_finger','A'),('map_arterial','B'),('glucose_lab','A'),('map_cuff','B')]:
    idx_,Xl=lagX(evn,wh); er=(R.value.values-R.mean_rec.values)[idx_]
    keep=Xl.std(0)>0; Xd=np.column_stack([np.ones(len(idx_)),Xl[:,keep]]); cf=np.linalg.lstsq(Xd,er,rcond=None)[0]; rs=er-Xd@cf
    se=np.sqrt(np.diag(np.linalg.pinv(Xd.T@Xd))*rs.var()); KAP[evn]=cf[1:]
    print(evn,'n',len(idx_),'coef',np.round(cf[1:],2),'se',np.round(se[1:],2),'var red %',np.round(100*(1-rs.var()/er.var()),2))
```