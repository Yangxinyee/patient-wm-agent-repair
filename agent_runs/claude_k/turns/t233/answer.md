Run 216 shows drug bundles never contain readings: they are action-only bundles. Their observation input is therefore empty, and the predicted rate can only come from `ctx`, the state before the bundle.

From ctx alone, cross-validated R² is 0.73 for the insulin rate but only 0.57 for the norepinephrine rate. That is good for insulin. For norepinephrine it is borderline, since an imprecise prediction leaks predictable dose variation into the innovation.

**Next run: build the predicted-rate decoders on more data.**
- Add the 300 extra train stays whose original-model states I fetched earlier (fwd 5–7) to the 99-stay set S2.
- Compute ctx at every drug bundle and fit ridge decoders for both rates.
- Save the weights and innovations to `/work`.

The plan is supported if R² holds up with more data and the innovations have reasonable spread. If norepinephrine stays near 0.57, its innovation term will carry extra noise. I would then shrink that channel's gain or decode with richer features.

```python
import json
Sx=[S2]+[np.load(f'/work/fwd_{j}/states_original.npz') for j in (5,6,7)]
XA_=np.concatenate([s['x'] for s in Sx]); stA=np.concatenate([s['stay_id'] for s in Sx]).astype(int); hrA=np.concatenate([s['hour'] for s in Sx])
dIA=np.concatenate([s['has_insulin'] for s in Sx])>0; dNA=np.concatenate([s['has_norepinephrine'] for s in Sx])>0
print('bundles',len(stA),'stays',len(np.unique(stA)),'in train csv',np.isin(np.unique(stA),tr_raw.stay_id.unique()).mean())
dr=np.where(dIA|dNA)[0]; dr=dr[dr>0]; dr=dr[stA[dr]==stA[dr-1]]
sub=tr_raw[tr_raw.stay_id.isin(set(np.unique(stA)))]
ki=sub[sub.event=='insulin'].groupby(['stay_id','hour'])['rate'].sum(); kn=sub[sub.event=='norepinephrine'].groupby(['stay_id','hour'])['rate'].sum()
riA=np.nan_to_num(get(ki,stA[dr],hrA[dr])); rnA=np.nan_to_num(get(kn,stA[dr],hrA[dr]))
def ctx_X(Xa,rows,dts):
    xr=Xa[rows]; o=xr[:,:192]; h=xr[:,192:960].reshape(-1,4,192)*np.exp(-dts[:,None,None]*fl[None,None,:])
    gg=gelu_np(np.log(dts)[:,None]/10*g0w[:,0][None,:]+g0b)@g2w.T+g2b
    return o+LNnp(h.mean(1),onw,onb)+gg
dtA=np.maximum((hrA[dr]-hrA[dr-1])*3600.0,60.0); CTA=ctx_X(XA_,dr-1,dtA)
from sklearn.linear_model import Ridge
grp=stA[dr]; ust=np.unique(grp); fold=np.searchsorted(ust,grp)%5
dec={}
for nmv,y_ in [('ins',riA),('ne',rnA)]:
    for al in [1.0,10.0,100.0]:
        p_=np.zeros_like(y_)
        for f in range(5):
            m_=Ridge(alpha=al).fit(CTA[fold!=f],y_[fold!=f]); p_[fold==f]=m_.predict(CTA[fold==f])
        print(nmv,'alpha',al,'grouped cvR2',np.round(1-((y_-p_)**2).mean()/y_.var(),3),'innov sd',np.round((y_-p_).std(),3),'rate sd',np.round(y_.std(),3))
    m_=Ridge(alpha=10.0).fit(CTA,y_); dec[nmv]=(m_.coef_.copy(),float(m_.intercept_))
np.savez('/work/decoders.npz',wA=dec['ins'][0],bA=dec['ins'][1],wB=dec['ne'][0],bB=dec['ne'][1])
print('n drug bundles',len(dr))
```