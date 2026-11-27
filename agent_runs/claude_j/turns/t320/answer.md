The core edit is now clean in the sense that matters: one-step drug-I and drug-N channels whose probes give near-zero output on other tokens.

But at the chosen scales it disturbs the factual predictions more than the linear analysis said, about +0.002 nats overall. The glucose-lab readings are the most sensitive (+0.0045), and part of this comes from probe noise on bundles without the drug.

The readout does not need a high SNR. The head unit's gain will be modest, about 1–3 glucose units per unit of rate, so an SNR of about 3 per unit rate adds negligible noise. I'll therefore cut the channel scales to an SNR of 3 for drug I and 6 for drug N. The disturbance should fall roughly quadratically.

In the same run I'll:
- regenerate the replica data on 100 stays;
- fit the head readouts, for drug I on glucose readings and drug N on MAP readings. The target is the latest dose deviation from its context-predicted value, with that expectation fitted on pre-dose states and cross-validated.
- check that the readout responds to x2 doses with slope ≈ 1;
- estimate the factual-optimal gains from model residuals.

I'll also print the names `heads_full` uses, so I know which weight dict the head edit must go into.

The approach is confirmed if all of these hold:
- disturbance below about 0.0005 nats;
- readout CV R² of at least 0.5;
- x2 response slope near 1;
- residual slopes negative for drug I on glucose and positive for drug N on MAP.

```python
from sklearn.model_selection import cross_val_predict
tI=TC@V2_[:,-1]*(3/np.sqrt(l2[-1])); tN=TC@V2_[:,-2]*(6/np.sqrt(l2[-2]))
BinF=Bin0.clone(); BinF+=torch.outer(torch.as_tensor(tI,dtype=dty),torch.as_tensor(rI,dtype=dty))+torch.outer(torch.as_tensor(tN,dtype=dty),torch.as_tensor(rN,dtype=dty))
t0=time.time()
try:
    for s in stays:
        rt=RTs[s]; eps_=D3[s]['eps'] if 'eps' in D3[s] else DAT[s]['eps']
        W[KV]=WkvF; W[BI]=BinF; O1,H1,_=run_cf(s,{},BUN[s][1][0],X_of(s)); D3[s]['wc1'],D3[s]['lg1']=hf2(O1,H1,rt,eps_)
        Xi=torch.cat([T64(O1),T64(H1).reshape(len(O1),-1),T64(X_of(s))[:,960:992]],1)
        for dr in ('insulin','norepinephrine'):
            Oc,Hc,_=run_cf(s,{dr:2.0},720,Xi); D3[s]['cf'][dr]=hf2(Oc,Hc,rt,eps_)
finally: W[KV]=Wkv0; W[BI]=Bin0
dm=np.concatenate([mean_of(D3[s]['lg1'],D3[s]['ev'])-mean_of(D3[s]['lg0'],D3[s]['ev']) for s in stays])
dn=np.concatenate([bnll(D3[s]['lg1'],D3[s]['ev'],D3[s]['ob'])-bnll(D3[s]['lg0'],D3[s]['ev'],D3[s]['ob']) for s in stays])
print('time',round(time.time()-t0,1),'dnll all',round(float(dn.mean()),5),[(e_[:6]+e_[-3:],round(float(np.sqrt((dm[EV==e_]**2).mean())),3),round(float(dn[EV==e_].mean()),5)) for e_ in np.unique(EV)])
CTX=[];RD=[];KEY=[]
for s in stays:
    B=BUN[s]; O0=D3[s]['O0']; H0=D3[s]['H0']
    for k in range(1,len(B)):
        evs=B[k][1]
        if any(a[0] in ('insulin','norepinephrine') for a in evs):
            CTX.append(np.r_[O0[k-1],H0[k-1]]); RD.append([sum(a[2].rate for a in evs if a[0]==d and a[1]=='start') for d in ('insulin','norepinephrine')]); KEY.append((s,k))
CTX=np.array(CTX); RD=np.array(RD); EXP=np.stack([cross_val_predict(Ridge(alpha=30.),CTX,RD[:,j],cv=5) for j in range(2)],1)
print('ctx R2',[round(1-((RD[:,j]-EXP[:,j])**2).sum()/((RD[:,j]-RD[:,j].mean())**2).sum(),3) for j in range(2)])
DV={kk:(RD[i],RD[i]-EXP[i]) for i,kk in enumerate(KEY)}
rows=[]
for s in stays:
    B=BUN[s]; mins=np.array([b[0] for b in B],float); rt=RTs[s]; isd=np.array([(s,k) in DV for k in range(len(B))])
    for i,(mn,b) in enumerate(zip(rt['minute'].values,rt['b'].values.astype(int))):
        ks=np.where(isd[:b])[0]; k=ks[-1] if len(ks) else -1; ok=k>=0 and (mn-mins[k])<=70
        r,dv=DV[(s,k)] if ok else (np.zeros(2),np.zeros(2)); post=ok and mins[k]>=720
        rows.append((s,i,D3[s]['ev'][i],r[0],r[1],dv[0],dv[1],post))
import pandas as pd
RW=pd.DataFrame(rows,columns=['s','i','ev','rI','rN','dI','dN','post'])
WC1=np.concatenate([D3[s]['wc1'] for s in stays]); WCc={dr:np.concatenate([D3[s]['cf'][dr][0] for s in stays]) for dr in ('insulin','norepinephrine')}
PM1=np.concatenate([mean_of(D3[s]['lg1'],D3[s]['ev']) for s in stays]); OB=np.concatenate([D3[s]['ob'] for s in stays])
GR={'insulin':(['glucose_finger','glucose_lab'],'dI','rI'),'norepinephrine':(['map_arterial','map_cuff'],'dN','rN')}; RO={}
for dr,(cods,dc,rc) in GR.items():
    m_=RW['ev'].isin(cods).values; y=RW[dc].values[m_]; Xw=WC1[m_]
    pcv=cross_val_predict(Ridge(alpha=1.0),Xw,y,cv=5); mdl=Ridge(alpha=1.0).fit(Xw,y); RO[dr]=mdl
    r2=1-((pcv-y)**2).sum()/((y-y.mean())**2).sum(); po=RW['post'].values[m_]
    du=mdl.predict(WCc[dr][m_])-mdl.predict(Xw); sl=np.polyfit(RW[rc].values[m_][po],du[po],1)[0]
    u=mdl.predict(Xw); res=OB[m_]-PM1[m_]; A=np.polyfit(u,res,1)[0]
    for c_ in cods:
        mc=RW['ev'].values[m_]==c_; print(dr[:3],c_[-6:],'cvR2',round(r2,3),'x2 slope',round(float(sl),3),'resid slope',round(float(np.polyfit(u[mc],res[mc],1)[0]),2))
print(heads_full.__code__.co_names[:20])
```