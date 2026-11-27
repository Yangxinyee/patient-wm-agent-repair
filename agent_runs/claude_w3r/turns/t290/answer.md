The 40-unit random-feature decoder fails inside the real path. It amplifies the stage-1 decode noise (FFN-output error 0.74 for insulin, 0.35 for norepinephrine), and with λ=15 the gate leaks (norepinephrine output up to 28 at u-absent bundles).

I switch to a more robust form: per drug, a gated linear decode from one GELU pair, pre± = λ(g−1) + B ± κ·L(reads). In the linear regime the difference is exactly 2κL. Gate noise cancels inside the pair, and at no-action both units are off. The cost is the accuracy of a linear decode from the two reads, so I check its bias by rate range, then build it and verify gating and noise.

```python
DECL={}
for nm,cc,y in [('ins',[1,2],yi),('ne',[3,4],yn)]:
    X_=np.hstack([Vu2[:,cc],np.ones((len(Vu2),1))]); a=np.linalg.lstsq(X_,y,rcond=None)[0]; pr=X_@a
    Lna=np.hstack([Vl0.reshape(-1,5)[:,cc]+RW2[192,cc],np.ones((len(ub),1))])@a
    DECL[nm]=dict(cols=cc,a=a,Lmax=float(np.abs(pr).max()),Lnoact=float(np.abs(Lna[~ub]).max())); print(nm,'linear dec',summ(pr,y),'Lmax',round(DECL[nm]['Lmax'],1),'L garbage u-absent max',round(DECL[nm]['Lnoact'],2))
W1n=F1w.numpy().copy(); b1x=F1b.numpy().copy(); W2n=F2w.numpy().copy()
for i,(nm,dd) in enumerate(DECL.items()):
    kap=10./dd['Lmax']; B=13.; lam=B+kap*dd['Lnoact']+6.; dd.update(kap=kap,B=B,lam=lam,units=UNITS[2*i:2*i+2])
    cc=dd['cols']; aR=dd['a'][:2]; a0=dd['a'][2]+aR@RW2[192,cc]
    lin_w=sum(aR[k]*RHO[:192,f] for k,f in enumerate(cc)); lin_b=sum(aR[k]*RHO[192,f] for k,f in enumerate(cc))+a0
    for sgn,uj in zip([1,-1],dd['units']):
        W1n[uj]=lam*RHO[:192,0]+sgn*kap*lin_w; b1x[uj]=lam*(RHO[192,0]-1)+B+sgn*kap*lin_b; W2n[:,uj]=sgn*DOUT[nm]/(2*kap)
E1,Hu=prelude_out(W2n); Eh=np.concatenate([E1,np.ones(E1.shape[:2]+(1,))],-1).reshape(-1,193); Hf=Hu.reshape(-1,768)
for nm,dd in DECL.items():
    cc=dd['cols']; ideal=(np.hstack([Vt[:,cc]+RW2[192,cc],np.ones((len(Vt),1))])@dd['a'])*ub
    actual=(Hf[:,dd['units'][0]]-Hf[:,dd['units'][1]])/(2*dd['kap']); vr=np.linalg.lstsq(Eh,actual,rcond=None)[0]; rd=Eh@vr; dd['vE']=vr
    print(nm,'lam',round(dd['lam'],1),'FFN-out vs ideal (u pres) sd',round(float((actual-ideal)[ub].std()),3),'bias',round(float((actual-ideal)[ub].mean()),3),'| E-read noise',round(float((rd-actual).std()),4),'| u-absent actual max',round(float(np.abs(actual[~ub]).max()),4),'read max',round(float(np.abs(rd[~ub]).max()),3))
```