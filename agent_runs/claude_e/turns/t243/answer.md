That error was only because the failed run never defined `W0c`. `RA_` carries stay and hour for the same rows, so I can recover the previous insulin rate from `train.csv` myself. I will redo the gate/readout split of units 26 and 41 and tabulate it by previous insulin rate on glucose_finger rows. For the readout term I will also check its slope against dev1 within each rate band. If the gate term drops or the slope flattens at low rates, that explains the weak x1/2 response there.

```python
W0c=load('attr_heads.value_bin.0.weight').numpy().astype(np.float64); b0c=load('attr_heads.value_bin.0.bias').numpy().astype(np.float64)
W2c=load('attr_heads.value_bin.2.weight').numpy().astype(np.float64)
pre=np.einsum('drk,uk->dru',WCall.astype(np.float64),W0c[[26,41]])+b0c[[26,41]]
gsum=(pre[...,0]+pre[...,1])/2; gdif=(pre[...,0]-pre[...,1])/2
assert (RA_.stay.values==A_.stay.values).all()
trI=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate'])
trI=trI[(trI.event=='insulin')&(trI.stay_id.isin(np.unique(A_.stay.values)))]
gI={s:(g.hour.values,g.rate.values) for s,g in trI.groupby('stay_id')}
hrs=RA_.hour.values; sts=A_.stay.values; prev_rate=np.full(len(A_),np.nan)
for i in np.where(A_.hi.values)[0]:
    if sts[i] in gI:
        hh_,rr_=gI[sts[i]]; k_=hh_<hrs[i]
        if k_.any(): prev_rate[i]=rr_[k_][np.argmax(hh_[k_])]
gf=(A_.code.values==6)
for hv in [False,True]:
    mk_=gf&(A_.hi.values==hv); print('hi',hv,'n',mk_.sum(),'gsum q05/50/95',np.round(np.quantile(gsum[:,mk_].mean(0),[.05,.5,.95]),2))
mk_=gf&A_.hi.values&~np.isnan(prev_rate)
T_=pd.DataFrame({'r':prev_rate[mk_],'gs':gsum[:,mk_].mean(0),'gd':gdif[:,mk_].mean(0),'dev1':A_.dev1.values[mk_]})
T_['rb']=pd.cut(T_.r,[0,.5,1,1.5,2,3,4,6,20])
def slope(g): 
    x=g.dev1-g.dev1.mean(); return (x*(g.gd-g.gd.mean())).sum()/max((x*x).sum(),1e-9)
print(T_.groupby('rb',observed=True).apply(lambda g: pd.Series({'n':len(g),'gs':g.gs.mean(),'gd':g.gd.mean(),'dev1':g.dev1.mean(),'slope_gd_dev1':slope(g)})).round(3).to_string())
```