Adding the last-bin one-hot doesn't help: R² stays around 0.23 and the counterfactual slope around 0.29. So about 29% of a true dose change shows up in the best linear readout.

Redoing the cost estimate with these actual numbers is less pessimistic than before. The gain needed is roughly 2.65 / 0.29 ≈ 9 mg/dL per unit of decoded innovation. That adds about 2–3 mg² to the finger-glucose squared error, roughly 0.002–0.003 nats overall, within the 0.005 limit. It is borderline but feasible, provided the real network responds to large dose changes the way the factual variation suggests. Only a dose_edit pass can confirm that.

Before building it I need the same numbers for norepinephrine on map_arterial. I'll fit the titration rule non-parametrically on the training stays: the expected rate given the previous rate and the hour's MAP. Then I'll take the innovation as recorded minus expected, and test its decodability and slope from the head query at arterial MAP readings.

```python
P=Ptr.copy(); P['M']=P.map_arterial.fillna(P.map_cuff); P['R']=P.norepinephrine.fillna(0)
P['Rp']=P.groupby(level=0)['R'].shift(1).fillna(0)
P['mb']=(P.M//2)*2
on_=(P.R>0)|(P.Rp>0)
cont=P[(P.Rp>0)&P.M.notna()]
ratio_tab=(cont.R/cont.Rp).groupby(cont.mb).mean()
start=P[(P.Rp==0)&P.M.notna()]
start_tab=start.R.groupby(start.mb).mean()
nomap_cont=P[(P.Rp>0)&P.M.isna()]; ratio_nomap=(nomap_cont.R/nomap_cont.Rp).mean()
def ne_expect(Pd):
    Pd=Pd.copy(); Pd['M']=Pd.map_arterial.fillna(Pd.map_cuff); Pd['R']=Pd.norepinephrine.fillna(0)
    Pd['Rp']=Pd.groupby(level=0)['R'].shift(1).fillna(0); Pd['mb']=(Pd.M//2)*2
    rt=Pd.mb.map(ratio_tab).fillna(1.0); st=Pd.mb.map(start_tab).fillna(0.0)
    ex=np.where(Pd.Rp>0, np.where(Pd.M.notna(), Pd.Rp*rt, Pd.Rp*ratio_nomap), np.where(Pd.M.notna(), st, 0.0))
    ex=np.minimum(ex,1.0); Pd['Rhat']=ex; Pd['nuN']=Pd.R-Pd.Rhat; return Pd
Ptr3=ne_expect(Ptr); Ptu3=ne_expect(Ptu)
m_=(Ptr3.R>0)|(Ptr3.Rp>0)
print('NE policy R2 (train, active hrs)',round(1-np.var(Ptr3.nuN[m_])/np.var(Ptr3.R[m_]),3),'sd nu',round(Ptr3.nuN[m_].std(),3))
mA=(fa.event=='map_arterial').values
subA=fa[mA].copy()
subA['nuN1']=[Ptu3.nuN.get((s_,t-1),0) for s_,t in zip(subA.stay_id,subA.hour)]
subA['R1']=[Ptu3.R.get((s_,t-1),0) for s_,t in zip(subA.stay_id,subA.hour)]
grp_=subA.stay_id.values
for nm,F in [('q',q_[mA]),('view',view_[mA])]:
    r1,pr1=cvr(F,subA.nuN1.values.astype(float)); r2,_=cvr(F,subA.R1.values.astype(float))
    print(nm,'R2 nuN1',r1,'slope',round(np.polyfit(subA.nuN1.values,pr1,1)[0],3),'R2 R1',r2, 'n',len(subA), 'frac on',round((subA.R1>0).mean(),3))
```