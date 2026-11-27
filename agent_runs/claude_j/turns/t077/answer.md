Lagged insulin rates can be read from the state with cross-validated R² from 0.8 (just given) down to 0.6 (four hours back). Norepinephrine rates read at about 0.5. These numbers are inflated by confounding, though: doses track glucose and MAP levels, which the state certainly encodes. What an edit needs is the part of the dose that is not explained by the reading it was chosen from (the random multiplier). Next I'll test whether that dose surprise (rate minus the scale's expected rate for the glucose reading at that hour) can be decoded from the state. High out-of-sample R² for the first few lags would mean the state already carries the information an edit can amplify. R² near 0 would mean the model largely ignores the surprise, and a head-only edit could not respond to dose changes.

```python
print(mean_by_base)
gfv = tu[tu.event=='glucose_finger'][['stay_id','hour','value']]
GFH = {(a,int(round(b))):v for a,b,v in zip(gfv.stay_id,gfv.hour,gfv.value)}
def base_lvl(g):
    if g is None or not np.isfinite(g) or g<=150: return 0
    return 1 if g<=180 else 2 if g<=220 else 3 if g<=260 else 4 if g<=300 else 6
mb_ = {0:0.0}; mb_.update({int(k):float(v) for k,v in mean_by_base.items()})
def surprise_lag(K=5):
    out_ = np.zeros((len(sid_),K))
    for i,(s,b) in enumerate(zip(sid_,base_h)):
        for k in range(K):
            d = HI.get((s,b-k),0.0); g = GFH.get((s,b-k),np.nan)
            out_[i,k] = d - mb_.get(base_lvl(g),0.0) if (d>0 or base_lvl(g)>0) else 0.0
    return out_
YS = surprise_lag()
print('surprise sd by lag', np.round(YS[keep].std(0),3))
for a_ in [10,100,1000]:
    print(a_, 'INS-surprise R2', np.round(cvr2(F_all,YS,a_),3))
```