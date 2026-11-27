Run 217 fitted the predicted-rate decoders on 8,707 drug bundles from 396 train stays. Stays were grouped in cross-validation.
- **Insulin:** R² 0.70, innovation sd 0.66 against a rate sd of 1.20.
- **Norepinephrine:** R² 0.58, innovation sd 0.23 against a rate sd of 0.36.

The norepinephrine innovation still contains dose variation that the records predict but ctx does not capture linearly.

**Next run: measure the actual error pattern.** On the 99 stays with tool-predicted means, I will regress the model's factual errors on lagged cross-fitted innovations:
- glucose_finger errors on insulin innovations;
- map_arterial errors on norepinephrine innovations.

This gives the factual-optimal coefficient per lag and the error-variance reduction available. I'll also check whether readings in the same hour come before or after the drug bundle.

The plan is supported if the lag-1 and lag-2 coefficients are clearly negative for insulin and positive for norepinephrine, with a measurable variance reduction. That would mean an innovation write can improve factual fit while adding the missing effect. Coefficients near zero would mean the ctx-innovation is too diluted to carry the effect.

```python
innA=np.zeros(len(dr)); innB=np.zeros(len(dr))
for f in range(5):
    tr_f=fold!=f; te_f=fold==f
    innA[te_f]=riA[te_f]-Ridge(alpha=10.0).fit(CTA[tr_f],riA[tr_f]).predict(CTA[te_f])
    innB[te_f]=rnA[te_f]-Ridge(alpha=1.0).fit(CTA[tr_f],rnA[tr_f]).predict(CTA[te_f])
# order check: readings vs drug bundles at same hour (S2 part only)
is2=dr<len(S2['stay_id']); drow2=set(zip(stA[dr][is2],hrA[dr][is2])); rb=R['k'].values+1
same=[(rb[i] < dr[is2][np.where((stA[dr][is2]==R.stay_id.values[i])&(hrA[dr][is2]==R.hour.values[i]))[0][0]]) for i in range(len(R)) if (R.stay_id.values[i],R.hour.values[i]) in drow2][:3000]
print('same-hour pairs',len(same),'reading before drug frac',np.round(np.mean(same),3))
from collections import defaultdict
bys=defaultdict(list)
for j in np.where(is2)[0]: bys[stA[dr[j]]].append((hrA[dr[j]],innA[j],innB[j]))
NL=8
def lagX(evn,which):
    idx_=np.where(R.event.values==evn)[0]; Xl=np.zeros((len(idx_),NL))
    for n_,i in enumerate(idx_):
        for (hb,a_,b_) in bys.get(R.stay_id.values[i],[]):
            lg_=R.hour.values[i]-hb
            if 1<=lg_<=NL: Xl[n_,int(np.ceil(lg_))-1]+= (a_ if which=='A' else b_)
    return idx_,Xl
for evn,wh in [('glucose_finger','A'),('map_arterial','B'),('glucose_lab','A'),('map_cuff','B')]:
    idx_,Xl=lagX(evn,wh); er=(R.value.values-R.mean_rec.values)[idx_]
    Xd=np.column_stack([np.ones(len(idx_)),Xl]); cf=np.linalg.lstsq(Xd,er,rcond=None)[0]; rs=er-Xd@cf
    se=np.sqrt(np.diag(np.linalg.inv(Xd.T@Xd))*rs.var())
    print(evn,'n',len(idx_),'coef',np.round(cf[1:],2),'se',np.round(se[1:],2),'var red %',np.round(100*(1-rs.var()/er.var()),2))
```