The lag regression shows two things:
- **One-step channel:** the decoded channel responds only to the drug bundle in the last hour (the coefficients at later lags are about 0). It is effectively a one-step channel, which fits a_bar ≈ 0.37 per step.
- **Probe flaw:** R² is only 0.006 because the drug-I probe was trained only on drug-I bundles. It therefore fires uncontrolled on drug-N-only and END-only bundles, and that false signal created the apparent persistence.

A one-step channel is acceptable. The direct effect needed is mostly at lag 1, the model's own persistence through readings carries the rest, and I can calibrate the gain with the tool.

Next I'll:
1. Retrain both probes on all drug bundles. The target is that drug's START rate (0 if absent), across scalings of both drugs, so each probe ignores the other drug.
2. Make the two b_in readers cross-talk-free in the null space of the normalised slots.
3. Build the edit: drug I at SNR ≈ 10 per unit rate, drug N at ≈ 30 per unit rate.
4. Run the replica on all 100 stays: original factual, edited factual, and edited with x2 for each drug from hour 12. I'll store head inputs and logits for fitting the readout.

The core edit is confirmed if its factual disturbance is negligible (|ΔNLL| < 0.0005 per event type) and the channel gives a clean response under x2.

```python
R2d={'insulin':[], 'norepinephrine':[]}
for s in stays:
    for k,(mn,evs) in enumerate(BUN[s]):
        if not any(a[0] in R2d for a in evs): continue
        r={d:sum(a[2].rate for a in evs if a[0]==d and a[1]=='start') for d in R2d}
        for d in R2d:
            o=[x for x in R2d if x!=d][0]
            for (dd,m) in [(d,0.),(d,.5),(d,1.),(d,1.5),(d,2.),(o,0.),(o,2.)]:
                R2d[d].append((r[d]*(m if dd==d else 1.), ln1(bvd(mod_events(evs,{dd:m}))[1]).numpy()))
PR2={}
for d in R2d:
    y=np.array([x[0] for x in R2d[d]]); Xd=np.stack([x[1] for x in R2d[d]])
    Am=Xd.T@Xd+3.0*len(y)/len(OT)*OT.T@OT+1e4*np.outer(LUe,LUe)+1e-3*len(y)*np.eye(192); w=np.linalg.solve(Am,Xd.T@y); PR2[d]=w
    p_=Xd@w; print(d,'R2',round(1-((p_-y)**2).sum()/((y-y.mean())**2).sum(),3),'zero-rows sd',round(float(p_[y==0].std()),3),'oth sd',round(float((OT@w).std()),3))
wI2=torch.as_tensor(PR2['insulin'],dtype=dty); wN2=torch.as_tensor(PR2['norepinephrine'],dtype=dty)
Cn=np.stack([Qn.T@dEI,Qn.T@dEN],0); Rr=Qn@np.linalg.pinv(Cn)    # cols: readers with r_I.dEI=1,r_I.dEN=0 ...
rI=Rr[:,0]/cp; rN=Rr[:,1]/cp
tI=TC@V2_[:,-1]*(10/np.sqrt(l2[-1])); tN=TC@V2_[:,-2]*(30/np.sqrt(l2[-2]))
WkvF=Wkv0.clone(); WkvF[288:336]+=cp*(torch.outer(torch.as_tensor(pEI,dtype=dty),wI2)+torch.outer(torch.as_tensor(pEN,dtype=dty),wN2))
BinF=Bin0.clone(); BinF+=torch.outer(torch.as_tensor(tI,dtype=dty),torch.as_tensor(rI,dtype=dty))+torch.outer(torch.as_tensor(tN,dtype=dty),torch.as_tensor(rN,dtype=dty))
print('|tI|',round(float(np.linalg.norm(tI)),1),'|tN|',round(float(np.linalg.norm(tN)),1))
t0=time.time(); D3={}
try:
    for s in stays:
        rt=RTs[s]; ev=rt['event'].values; ob=rt['observed'].values; eps_=DAT[s]['eps']
        W[KV]=Wkv0; W[BI]=Bin0; O0,H0,_=run_cf(s,{},BUN[s][1][0],X_of(s)); wc0,lg0=hf2(O0,H0,rt,eps_)
        W[KV]=WkvF; W[BI]=BinF; O1,H1,_=run_cf(s,{},BUN[s][1][0],X_of(s)); wc1,lg1=hf2(O1,H1,rt,eps_)
        Xi=torch.cat([T64(O1),T64(H1).reshape(len(O1),-1),T64(X_of(s))[:,960:992]],1); cf={}
        for dr in ('insulin','norepinephrine'):
            Oc,Hc,_=run_cf(s,{dr:2.0},720,Xi); cf[dr]=hf2(Oc,Hc,rt,eps_)
        D3[s]=dict(ev=ev,ob=ob,wc0=wc0,lg0=lg0,wc1=wc1,lg1=lg1,cf=cf,O0=O0.numpy(),H0=H0.mean(1).numpy())
finally: W[KV]=Wkv0; W[BI]=Bin0
dm=np.concatenate([mean_of(D3[s]['lg1'],D3[s]['ev'])-mean_of(D3[s]['lg0'],D3[s]['ev']) for s in stays])
dn=np.concatenate([bnll(D3[s]['lg1'],D3[s]['ev'],D3[s]['ob'])-bnll(D3[s]['lg0'],D3[s]['ev'],D3[s]['ob']) for s in stays])
print('time',round(time.time()-t0,1)); print([(e_[:6],round(float(np.sqrt((dm[EV==e_]**2).mean())),3),round(float(dn[EV==e_].mean()),5)) for e_ in np.unique(EV)])
```