The head-input rebuild is now exact (z prior and with_code reproduce to 0.0). The probe's footprint at map_arterial readings is uneven: mean |Δwc| 0.38 but p90 2.0, against a natural spread of about 0.26 per dimension (about 3.6 in norm). So it is large at some readings, presumably just after injected NE bundles, and small elsewhere.

Next I relate Δwc at map_arterial readings to the most recent NE START bundle before the reading: its injected amount q = 0.5 + 0.5·rate and the minutes since it. I also check whether the response direction is consistent across readings, because a linear reader needs one dominant direction. A strong, consistent footprint at the first reading after an NE start, decaying with time, would confirm the carrier works.

```python
TRn=pd.read_csv('/data/train.csv'); TRn=TRn[TRn.stay_id.isin(selP)&(TRn.event=='norepinephrine')&TRn.rate.notna()]
ns={s:(np.round(g.hour.values*60).astype(int),0.5+0.5*g.rate.values) for s,g in TRn.groupby('stay_id')}
mins=RD.minute.values[ri]; sids=RD.stay_id.values[ri]
q1=np.zeros(len(ri)); ms=np.full(len(ri),1e9); q2=np.zeros(len(ri))
for i,(s,mn) in enumerate(zip(sids,mins)):
    if s not in ns: continue
    t,q=ns[s]; k=np.where(t<mn)[0]
    if len(k): q1[i]=q[k[-1]]; ms[i]=mn-t[k[-1]]; q2[i]=q[k[-2]] if len(k)>1 else 0
A_=(ev=='map_arterial')
for lo,hi in [(0,30),(30,60),(60,90),(90,180),(180,1e8)]:
    m=A_&(ms>=lo)&(ms<hi); print(f'since[{lo},{hi}) n={m.sum()} mean|dwc|={nd[m].mean():.3f} corr(|dwc|,q1)={np.corrcoef(nd[m],q1[m])[0,1] if m.sum()>5 else np.nan:.2f}')
m=A_&(ms<90); Dm=dW[m]; mu=Dm.mean(0); cs=(Dm@mu)/(np.linalg.norm(Dm,axis=1)*np.linalg.norm(mu)+1e-9)
print('lag1 (<90min) MAP_a: |mean dwc|',np.linalg.norm(mu).round(3),'median cos to mean',np.median(cs).round(2),'frac cos>.5',(cs>.5).mean().round(2))
U_,S_,V_=np.linalg.svd(Dm,full_matrices=False); print('dwc SV share', (S_[:5]**2/np.sum(S_**2)).round(3))
```