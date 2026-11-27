Result: shifting the glucose prediction by the full effect level would cost 0.44 nats per reading. Shifting by the increment since the last reading of the same code would cost 0.017, well over the 0.005 limit. The jitter-deviation increment would improve the fit by 0.0015.

So the edit must be band-relative, at least in part. The lab cost (0.12 per lab reading) is probably overstated, though, because I measured the increment from the last lab reading only. The model can also anchor on the finger readings.

Next I'll recompute the increment from the last glucose reading of either kind, for two signals:
- the full dose;
- the dose minus a constant reference, applied only when insulin runs.

A constant reference would be far easier to wire into the network than a band-dependent one. If some constant reference keeps the cost at or under about 0.003, I can skip band compensation. Otherwise I need the state's band encoding.

```python
gg2 = gg.sort_values(['stay_id','hour']).copy(); gg2['tlast']=gg2.groupby('stay_id').ih.shift(1)
gg2.loc[gg2.tlast==gg2.ih,'tlast']=np.nan
gg2['tlast']=gg2.groupby('stay_id').tlast.ffill() if False else gg2['tlast']
def cost(Amat, sub_all):
    tot=0.0
    for ev_,sub_ in sub_all.groupby('event'):
        m_=sub_.predicted_mean.values; y_=sub_.observed.values
        c_=np.median(np.abs(y_-m_)/m_)/0.6745; sd_=c_*m_
        e_=edges[ev_]; bi=np.searchsorted(e_,y_,side='right'); lo=np.r_[-np.inf,e_][bi]; hi=np.r_[e_,np.inf][bi]
        sh=np.array([lev(Amat,ix[s_],t_)-(lev(Amat,ix[s_],int(tl)) if np.isfinite(tl) else 0.0) for s_,t_,tl in zip(sub_.stay_id,sub_.ih,sub_.tlast)])
        p0=N_.cdf((hi-m_)/sd_)-N_.cdf((lo-m_)/sd_); p1=N_.cdf((hi-m_-sh)/sd_)-N_.cdf((lo-m_-sh)/sd_)
        tot+=np.sum(-np.log(np.clip(p1,1e-12,1))+np.log(np.clip(p0,1e-12,1)))
    return tot/len(f2)
print('full', round(cost(R,gg2),4), 'dev', round(cost(RD,gg2),4))
for dref in [1.5,2.0,2.5,3.0]:
    print('dref',dref, round(cost(np.where(R>0,R-dref,0.0),gg2),4))
```