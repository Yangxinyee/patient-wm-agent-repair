The gate failed. The readout's raw value (before group centring) spans hundreds of units across codes and last bins, against a within-group spread of about 0.05, so the ±40 code gate was swamped. Two reasons:
- Centring left the readout unpenalised along between-group directions (code and bin embeddings).
- The gate itself picks up noise from the query.

The edit is currently saved but broken, and I will overwrite it.

Fix:
1. Project the readout off the code-embedding rows and the last_numeric columns, and also penalise its raw (uncentred) variance, so its range stays modest.
2. Build the gate orthogonal to the last_numeric columns and any bin tables, with minimum within-group variance.
3. Compute the separation needed and the extra tilt noise the gate injects.

A gate-noise ratio well under about 0.3, with little loss in fraction, means the design works. Otherwise I need a different gating mechanism.

```python
nb=[(n,tuple(SHAPES[n])) for n in NAMES if ('bin' in n) and ('attr_heads' not in n) and ('embed_transition' not in n)]; print('bin tensors',nb[:6])
BT=[L_(n) for n,s in nb if len(s)==2 and s[1]==192 and s[0]<=20]
Cm=np.vstack([EC,LNW.T]+BT); Pc=np.eye(192)-Cm.T@np.linalg.pinv(Cm.T)
gall=(rowsM.event+'_'+rowsM.ri.map(dict(zip(pd.concat([fac,faN]).ri,pd.concat([fac,faN]).lb))).fillna(-1).astype(int).astype(str)).values
Xm=WC[rowsM.ri.values]; Xw=Xm-pd.DataFrame(Xm).groupby(gall).transform('mean').values
Sw=np.cov(Xw.T)+1e-4*np.eye(192); Swi=np.linalg.inv(Sw)
def gate1(pos):
    t=np.array([1.0 if c in pos else -1.0 for c in range(13)]+[0.0]*(Cm.shape[0]-13)); return Swi@Cm.T@np.linalg.solve(Cm@Swi@Cm.T,t)
def fitw(fa_,c_,v_,rho,lam,lam2):
    c17c=np.stack([cenof(e) for e in c_.event])[:,None,:]; T=(c_.S-rho*c_.Sp-(c_.mean_cf-c_.mean_f)).values
    th=np.clip(tsolve(LG[c_.ri.values],v_,c17c,T),-1.5,1.5); Xc_=(WC[c_.ri.values]-WC[c_.fi.values])@Pc; Xf_=WC[fa_.ri.values]
    gk=(fa_.event+'_'+fa_.lb.astype(str)).values; st=np.array(sorted(fa_.stay.unique())); fo={s:i%5 for i,s in enumerate(st)}
    ff_=fa_.stay.map(fo).values; fc_=c_.stay.map(fo).values; thf=np.zeros(len(fa_)); thc=np.zeros(len(c_))
    def sol(tf,tc):
        gm=pd.DataFrame(Xf_[tf]).groupby(gk[tf]).mean(); G_=gm.reindex(gk).values; G_=np.where(np.isnan(G_),Xf_[tf].mean(0)[None,:],G_)
        Xft=(Xf_-G_)@Pc; Xfr=(Xf_-Xf_[tf].mean(0))@Pc
        A=Xc_[tc].T@Xc_[tc]+lam*Xft[tf].T@Xft[tf]+lam2*Xfr[tf].T@Xfr[tf]; A+=1e-3*np.trace(A)/192*np.eye(192)
        return Pc@np.linalg.solve(A,Xc_[tc].T@th[tc]), Xft
    for k in range(5):
        w,Xft=sol(ff_!=k,fc_!=k); thf[ff_==k]=Xft[ff_==k]@w; thc[fc_==k]=Xc_[fc_==k]@w
    w,_=sol(np.ones(len(fa_),bool),np.ones(len(c_),bool))
    m0,_=mstat(LG[c_.ri.values],np.zeros(len(c_)),v_,c17c); m1,_=mstat(LG[c_.ri.values],thc,v_,c17c); mn=((c_.h0==12)&(c_.m==2.0)).values
    fr=(m1-m0)[mn].mean()/T[mn].mean(); ar_=np.arange(len(fa_)); ob_=fa_.ob.values; n0=-np.log(mix(LG[fa_.ri.values],np.zeros(len(fa_)),v_).mean(1)[ar_,ob_])
    cst=[round(float((-np.log(np.maximum(mix(LG[fa_.ri.values],g*thf,v_).mean(1)[ar_,ob_],1e-12))-n0).mean()),4) for g in [1.0,1/max(fr,1e-3)]]
    return w,fr,cst
for nm,fa_,c_,v_,rho,lam,pos in [('INS',fac,cf,vG,0.4,30,[6,10]),('NE',faN,cN,vM,0.3,300,[7,5])]:
    wg=gate1(pos); eta=Xm@wg-np.array([1.0 if c in pos else -1.0 for c in cidM]); sde=np.std(Xw@wg); Re=eta.max()-eta.min()
    for lam2 in [0.3,3.0]:
        w,fr,cst=fitw(fa_,c_,v_,rho,lam,lam2); r=Xm@w; Rr=r.max()-r.min(); sr=np.std(Xw@w)
        print(nm,'lam2',lam2,'frac',round(float(fr),3),'cost g1/full',cst,'Rr/sr',round(Rr/sr,1),'eta range',round(Re,3),'gate noise ratio',round(Rr/(2-Re)*sde/sr,3) if Re<2 else 'NA')
```