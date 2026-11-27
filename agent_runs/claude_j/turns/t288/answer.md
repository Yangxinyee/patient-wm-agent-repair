With the code-embedding directions also projected out, insulin's out-of-fold cost at gain 1 jumped to 0.13. Raising the raw penalty λ2 to 30 brings it to 0.004, but the fraction drops to 0.19. The earlier setup did much better: no code projection, ridge factor 1e-2, fraction 0.46 at cost 0.0018. So the culprit is probably the extra projection and/or the weaker ridge.

The code constants do not need projecting out. The gate's per-code targets can absorb them.

A third unit can cancel the gate noise. Dead unit 149 gets the same gate input and opposite output, so for insulin only the readout term survives.

Next run:
1. Re-test the readout with only the last_numeric projection and ridge factors 1e-2 and 1e-1, reporting its spread over glucose rows and over MAP rows.
2. Scan the original head for other near-dead hidden units, so drug B could also get a cancelling unit.

If the 1e-2 setting brings back fraction ≈ 0.46 at low cost, I build the gated, cancelled edit with it.

```python
H0o=load_original('attr_heads.value_bin.0.weight'); H0o=(H0o.numpy() if hasattr(H0o,'numpy') else np.asarray(H0o)).astype(np.float64)
b0o=np.asarray(load_original('attr_heads.value_bin.0.bias'),dtype=np.float64); H2o=np.asarray(load_original('attr_heads.value_bin.2.weight'),dtype=np.float64)
from scipy.special import erf
A_=WC[rowsM.ri.values]@H0o.T+b0o; Gl=0.5*A_*(1+erf(A_/np.sqrt(2))); contrib=np.abs(Gl).max(0)*np.abs(H2o).max(0)
o=np.argsort(contrib); print('weakest units',[(int(j),round(float(contrib[j]),4)) for j in o[:8]])
def fitw2(fa_,c_,v_,rho,lam,lam2,P_,rf):
    global Pc
    Pc=P_; c17c=np.stack([cenof(e) for e in c_.event])[:,None,:]; T=(c_.S-rho*c_.Sp-(c_.mean_cf-c_.mean_f)).values
    th=np.clip(tsolve(LG[c_.ri.values],v_,c17c,T),-1.5,1.5); Xc_=(WC[c_.ri.values]-WC[c_.fi.values])@P_; Xf_=WC[fa_.ri.values]
    gk=(fa_.event+'_'+fa_.lb.astype(str)).values; st=np.array(sorted(fa_.stay.unique())); fo={s:i%5 for i,s in enumerate(st)}
    ff_=fa_.stay.map(fo).values; fc_=c_.stay.map(fo).values; thf=np.zeros(len(fa_)); thc=np.zeros(len(c_))
    def sol(tf,tc):
        gm=pd.DataFrame(Xf_[tf]).groupby(gk[tf]).mean(); G_=gm.reindex(gk).values; G_=np.where(np.isnan(G_),Xf_[tf].mean(0)[None,:],G_)
        Xft=(Xf_-G_)@P_; Xfr=(Xf_-Xf_[tf].mean(0))@P_
        A=Xc_[tc].T@Xc_[tc]+lam*Xft[tf].T@Xft[tf]+lam2*Xfr[tf].T@Xfr[tf]; A+=rf*np.trace(A)/192*np.eye(192)
        return P_@np.linalg.solve(A,Xc_[tc].T@th[tc]), Xft
    for k in range(5):
        w,Xft=sol(ff_!=k,fc_!=k); thf[ff_==k]=Xft[ff_==k]@w; thc[fc_==k]=Xc_[fc_==k]@w
    w,_=sol(np.ones(len(fa_),bool),np.ones(len(c_),bool))
    m0,_=mstat(LG[c_.ri.values],np.zeros(len(c_)),v_,c17c); m1,_=mstat(LG[c_.ri.values],thc,v_,c17c); mn=((c_.h0==12)&(c_.m==2.0)).values
    fr=(m1-m0)[mn].mean()/T[mn].mean(); ar_=np.arange(len(fa_)); ob_=fa_.ob.values; n0=-np.log(mix(LG[fa_.ri.values],np.zeros(len(fa_)),v_).mean(1)[ar_,ob_])
    cst=[round(float((-np.log(np.maximum(mix(LG[fa_.ri.values],g*thf,v_).mean(1)[ar_,ob_],1e-12))-n0).mean()),4) for g in [1.0,1/max(fr,1e-3)]]
    return w,fr,cst
ipos=np.isin(cidM,[6,10])
for nm,fa_,c_,v_,rho,lam in [('INS',fac,cf,vG,0.4,30),('NE',faN,cN,vM,0.3,300)]:
    for rf,lam2 in [(1e-2,0.0),(1e-1,0.0),(1e-2,1.0)]:
        w,fr,cst=fitw2(fa_,c_,v_,rho,lam,lam2,Pn,rf); r=Xm@w; sr=np.std(Xw@w)
        print(nm,'rf',rf,'lam2',lam2,'frac',round(float(fr),3),'cost',cst,'R glu/sr',round(np.ptp(r[ipos])/sr,1),'R map/sr',round(np.ptp(r[~ipos])/sr,1))
```