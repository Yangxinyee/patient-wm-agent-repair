This decides the design. A decoder fitted directly on edited-minus-factual state differences explains only 55% of the needed correction change (slope 0.5). The same decoder applied to factual rows gives values with sd 46–197 mg/dL, against an sd of 1.25 for the real target. So the 192-number summary the value head sees does not hold a clean, separable dose signal. A correction placed only in the head would either miss most of the counterfactual effect or wreck factual fit.

Before choosing how to carry the dose into the state, I need to know where the dose information is lost. Is it absent from the core state altogether, or present in the 4 slots of h but blurred when they are averaged and normalised into `out`? I'll store the full state (out + 4 slots) before each reading. I'll fit decoders for the insulin rate at lags 1–3 from out alone, the slots alone and both, and report held-out R² and counterfactual slopes. High R² and slope ≈1 from the slots would mean the channel already exists and only needs routing to the head.

```python
code_roll3=code_roll2.replace("store.append((hn_.copy(),hj.copy(),base[inv].clone(),P.clone(),f8.clone(),lb_.clone(),zmu[inv].clone()))",
 "store.append((hn_.copy(),hj.copy(),base[inv].clone(),P.clone(),f8.clone(),lb_.clone(),zmu[inv].clone(),(h[un][inv].reshape(len(hj),-1).clone() if store_h else None),(out[un][inv].clone() if store_h else None)))")
code_roll3=code_roll3.replace("want_nll=False,chunk=None,store=None):","want_nll=False,chunk=None,store=None,store_h=False):")
open('/work/rollout.py','w').write(code_roll3); exec(code_roll3)
idsI2=idsI[:300]; RI2=prep_rollout(tr,idsI2); st2=[]
res2=run_rollout(RI2,W,armsI,n_paths=1,seed=0,t_hide=1e9,want_nll=True,store=st2,store_h=True)
hn2=np.concatenate([s[0] for s in st2]); hj2=np.concatenate([s[1] for s in st2])
H2=torch.cat([s[7] for s in st2]).numpy(); O2=torch.cat([s[8] for s in st2]).numpy()
A2_=RI2['A']; code2=A2_.code.values[hj2]; t2=A2_.t.values[hj2]; si2=hn2//nAI; arm2=hn2%nAI
Dm2=np.zeros((len(RI2['stays']),NH)); ia2=A2_[(A2_.code==CODE['insulin'])&(A2_.kind==0)]
Dm2[ia2.si.values,np.floor(ia2.t.values).astype(int)-H0]=ia2.rate.values
hh2=np.round(t2).astype(int)
def dl2(k):
    h_=hh2-k; d=np.where(h_-H0>=0,Dm2[si2,np.clip(h_-H0,0,NH-1)],0.0); return d*np.where(h_>=frm[arm2],fac[arm2],1.0), d
g2=np.isin(code2,[CODE['glucose_finger']]); tr2=(si2%2==0)
i0=pd.Series(np.arange(len(hn2))[arm2==0],index=pd.MultiIndex.from_arrays([si2[arm2==0],hj2[arm2==0]])).reindex(pd.MultiIndex.from_arrays([si2,hj2])).values
for nm_,Xf_ in [('out',O2),('slots',H2),('all',np.c_[O2,H2])]:
    res_l=[]
    for k in [1,2,3]:
        tg,tg0=dl2(k)
        rg=Ridge(alpha=1.0).fit(Xf_[g2&tr2],tg[g2&tr2]); dec=rg.predict(Xf_)
        r2=1-np.mean((tg[g2&~tr2]-dec[g2&~tr2])**2)/np.var(tg[g2&~tr2])
        mm_=g2&~tr2&(arm2>0)&(np.abs(tg-tg0)>1e-9)
        sl=np.polyfit(tg[mm_]-tg0[mm_],dec[mm_]-dec[i0[mm_]],1)[0]
        res_l.append((k,round(r2,3),round(sl,3)))
    print(nm_,res_l)
```