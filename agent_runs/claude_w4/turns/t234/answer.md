The states for both models on the 100 tuning stays are in. I'll now measure what the v3 injection actually did inside the model, in four steps.

1. **Direct effect.** I'll take each stay's first drug bundle, where nothing has been injected before, so the state change there is the direct effect. I'll check its size in `out` and in the slots, and whether it matches the stored Jacobian `Jm` applied to the injected vector p = −tI·dI + fN·dN. If it matches, I can use `J` to design a better direction; if it doesn't, `J` means something else.
2. **Persistence.** I'll check how the change carries over to the next bundles.
3. **Glucose readout.** I'll fit how the original model's predicted glucose mean depends on the `out` state before each reading, and project the injected change onto it. That tells me directly how many mg/dL per unit/h the carrier produces at the next reading.
4. **Alignment.** First I'll confirm that the drug bundles in the states line up with the cached tokens and rates (`Uk`, `RI`, `RN`).

```python
So=np.load('/work/fwd_7/states_original.npz'); Se=np.load('/work/fwd_7/states_edited.npz')
xo=So['x'].astype(np.float64); xe=Se['x'].astype(np.float64); sid=So['stay_id']; hr=So['hour']
hiB=So['has_insulin'].astype(bool); hnB=So['has_norepinephrine'].astype(bool); drug=hiB|hnB
print('rows',len(xo),'drug bundles',drug.sum(),'Uk rows',len(Uk), 'same as fwd_3 X3:', np.abs(xo-X3).max())
D=xe-xo
nO=np.linalg.norm(D[:,:192],axis=1); nH=np.linalg.norm(D[:,192:960].reshape(-1,4,192),axis=2).mean(1)
# map drug-bundle order to Uk/RI/RN
di=np.where(drug)[0]; tIk=LN1(Uk)@BB3['tI']; fNk=LN1(Uk)@BB3['fN']
print('RI>0 vs has_insulin agreement', np.mean((RI>0)==hiB[di]), ' RN>0 vs has_ne', np.mean((RN>0)==hnB[di]))
# first drug bundle per stay
first=[]; 
for s in np.unique(sid):
    w=np.where((sid==s)&drug)[0]
    if len(w): first.append(w[0])
first=np.array(first); kpos={r:i for i,r in enumerate(di)}
P_=np.stack([-tIk[kpos[r]]*dI + fNk[kpos[r]]*dN for r in first])
print('first drug bundles:',len(first),' mean |p| %.2f'%np.linalg.norm(P_,axis=1).mean())
# previous-bundle delta at first drug bundle should be ~0
prev_d=np.array([np.abs(D[r-1]).max() if r>0 and sid[r-1]==sid[r] else 0 for r in first]); print('max prev |D| before first drug', prev_d.max())
dy=D[first,192:960]; do=D[first,:192]
pred=P_@Jm.T
cs_=[float(a@b/np.linalg.norm(a)/np.linalg.norm(b)) for a,b in zip(dy,pred) if np.linalg.norm(b)>1e-9]
sc_=[float(a@b/(b@b)) for a,b in zip(dy,pred) if np.linalg.norm(b)>1e-9]
print('direct dy vs Jm@p: cos median %.3f IQR [%.3f,%.3f]; scale median %.3f'%(np.median(cs_),*np.percentile(cs_,[25,75]),np.median(sc_)))
print('direct |dout| per |p|: median %.4f ; |dh|/|p| median %.4f'%(np.median(np.linalg.norm(do,axis=1)/np.linalg.norm(P_,axis=1)), np.median(np.linalg.norm(dy.reshape(-1,4,192),axis=2).mean(1)/np.linalg.norm(P_,axis=1))))
# persistence: ratio of |D| at bundle after first drug (if non-drug) to |D| at first drug
nxt=[r+1 for r in first if r+1<len(sid) and sid[r+1]==sid[r] and not drug[r+1]]
rat_o=[nO[r]/max(nO[r-1],1e-12) for r in nxt]; rat_h=[nH[r]/max(nH[r-1],1e-12) for r in nxt]
print('persistence next non-drug bundle: |dout| ratio median %.3f, |dh| ratio median %.3f (n %d)'%(np.median(rat_o),np.median(rat_h),len(nxt)))
print('overall: median |dout| at drug %.4f non-drug %.4f; |dh| drug %.3f non-drug %.3f; out norm %.2f'%(np.median(nO[drug]),np.median(nO[~drug]),np.median(nH[drug]),np.median(nH[~drug]),np.linalg.norm(xo[:,:192],axis=1).mean()))
```