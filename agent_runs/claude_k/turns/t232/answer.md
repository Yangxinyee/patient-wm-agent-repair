Run 215 identified the two set encoders, `encode` for observations and `encode_action` for actions. `D2` is my own hourly analysis panel, not the raw record. To get each drug bundle's actual rates, I'll read the raw events of the 99 stays from `/data/train.csv`.

**Why this matters.** The writer must add the dose innovation, the rate minus what the model's inputs predict, not the raw dose. A raw-dose term would bias factual predictions by an amount set by the clinicians' dosing pattern, far above the 0.005-nat budget. The open question is whether the predicted rate needs the bundle's own observations (e) or whether the pre-bundle context alone suffices.

**Test.** On the S2 drug bundles I will fit the rate from ctx alone, and from ctx plus the same-bundle glucose or MAP reading, and compare cross-validated R². If the reading adds little, ctx-only innovations are enough. If it adds a lot, I must rebuild e.

```python
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
sset=set(int(s) for s in np.unique(S2['stay_id']))
tr_raw=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','value','rate'])
tr99=tr_raw[tr_raw.stay_id.isin(sset)]; print(tr99.event.unique()[:16])
dH=S2['has_insulin']>0; dN=S2['has_norepinephrine']>0; drows=np.where(dH|dN)[0]; drows=drows[drows>0]
drows=drows[S2['stay_id'][drows]==S2['stay_id'][drows-1]]
key=lambda ev: tr99[tr99.event.str.contains(ev)].groupby(['stay_id','hour'])['rate'].sum()
ki=key('insulin'); kn=key('norepi'); kg=tr99[tr99.event=='glucose_finger'].groupby(['stay_id','hour'])['value'].mean(); km=tr99[tr99.event=='map_arterial'].groupby(['stay_id','hour'])['value'].mean()
st_=S2['stay_id'][drows].astype(int); hh_=S2['hour'][drows]
def get(s_,stt,hrs): return np.array([s_.get((a,b),np.nan) for a,b in zip(stt,hrs)])
ri_=np.nan_to_num(get(ki,st_,hh_)); rn_=np.nan_to_num(get(kn,st_,hh_)); gv=get(kg,st_,hh_); mv=get(km,st_,hh_)
dtd=(S2['hour'][drows]-S2['hour'][drows-1])*3600.0; dtd=np.maximum(dtd,60.0)
CTd=ctx_vec(drows-1,dtd)
print('n drug bundles',len(drows),'ins>0',(ri_>0).mean().round(3),'ne>0',(rn_>0).mean().round(3),'gluc in bundle',np.isfinite(gv).mean().round(3),'map in bundle',np.isfinite(mv).mean().round(3))
def cvr2(Xf,y_): p_=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)),Xf,y_,cv=5); return 1-((y_-p_)**2).mean()/y_.var()
for nmv,y_,rv in [('ins',ri_,gv),('ne',rn_,mv)]:
    f_=np.isfinite(rv); rvf=np.where(f_,rv,0)
    print(nmv,'R2 ctx',np.round(cvr2(CTd,y_),3),' ctx+reading',np.round(cvr2(np.column_stack([CTd,f_,rvf,rvf**2/100]),y_),3))
```