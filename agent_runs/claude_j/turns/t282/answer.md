Shaping the target for the weak rollout feedback (ρ=0.4) makes the needed extra direct shift for insulin −6.4 per reading (main contrast). At the best λ (30), the out-of-fold readout reaches 46% at gain 1. Full gain costs about +0.011 NLL per glucose reading, about 0.0047 overall. Drug B at ρ=0.3 costs about 0.0056 per MAP reading at full gain, about 0.003 overall. Together that exceeds the 0.005 budget. The cost also grows faster than quadratically, so a few rows with large tilts dominate it.

Options to cut the cost:
- **Gate:** open the edit only when a drug bundle directly precedes the reading (gap 55 min), so rows without a recent dose add no noise.
- **Weighting:** down-weight rows where a tilt barely moves the mean.

I also need the exact head tensors and code indices to build the edit. This run:
- lists those tensors and indices;
- checks which weight dict the replica head functions read;
- splits the out-of-fold factual cost and the cf effect by whether a drug bundle preceded the reading.

If most of the cost sits on rows without a preceding drug bundle while the cf effect sits on rows with one, gating is the fix.

```python
vb=[n for n in NAMES if ('value_bin' in n) or n.startswith('embed.code') or ('last_numeric' in n)]
print([(n,tuple(SHAPES[n])) for n in vb])
for f_ in [logits_draws, with_code, value_probs, heads_full, q_draws]:
    co=f_.__code__; print(f_.__name__, co.co_varnames[:co.co_argcount], [x for x in co.co_names if x in ('W','Wt','load','load_original')])
cd=VOCAB.get('codes'); print('codes', type(cd).__name__, (list(cd.items())[:12] if isinstance(cd,dict) else cd[:12]))
def dtprev(df): return np.array([BUN[s][b][0]-BUN[s][b-1][0] if b>0 else -1 for s,b in zip(df.stay,df.b)])
for nm,fa_,c_,v_,rho,lam in [('INS',fac,cf,vG,0.4,30),('NE',faN,cN,vM,0.3,300)]:
    dtf=dtprev(fa_); dtc=dtprev(c_)
    c17c=np.stack([cenof(e) for e in c_.event])[:,None,:]; LGc_=LG[c_.ri.values]; LGf_=LG[fa_.ri.values]
    T=(c_.S-rho*c_.Sp-(c_.mean_cf-c_.mean_f)).values; th=np.clip(tsolve(LGc_,v_,c17c,T),-1.5,1.5)
    Xc_=WC[c_.ri.values]-WC[c_.fi.values]; Xf_=WC[fa_.ri.values]
    st=np.array(sorted(fa_.stay.unique())); fo={s:i%5 for i,s in enumerate(st)}; ff_=fa_.stay.map(fo).values; fc_=c_.stay.map(fo).values
    gk=(fa_.event+'_'+fa_.lb.astype(str)).values; thf=np.zeros(len(fa_)); thc=np.zeros(len(c_))
    for k in range(5):
        tf=ff_!=k; tc=fc_!=k; gm=pd.DataFrame(Xf_[tf]).groupby(gk[tf]).mean(); G=gm.reindex(gk).values
        G=np.where(np.isnan(G),Xf_[tf].mean(0)[None,:],G); Xft=Xf_-G
        A=Xc_[tc].T@Xc_[tc]+lam*Xft[tf].T@Xft[tf]; A+=1e-2*np.trace(A)/192*np.eye(192)
        w=np.linalg.solve(A,Xc_[tc].T@th[tc]); thf[~tf]=Xft[~tf]@w; thc[~tc]=Xc_[~tc]@w
    m0,_=mstat(LGc_,np.zeros(len(c_)),v_,c17c); m1,_=mstat(LGc_,thc,v_,c17c); ach=m1-m0
    ar_=np.arange(len(fa_)); ob_=fa_.ob.values; n0=-np.log(mix(LGf_,np.zeros(len(fa_)),v_).mean(1)[ar_,ob_])
    n1=-np.log(np.maximum(mix(LGf_,2.0*thf,v_).mean(1)[ar_,ob_],1e-12))-n0
    for lab,mk_f,mk_c in [('drug55',dtf==55,dtc==55),('other',dtf!=55,dtc!=55)]:
        print(nm,lab,'frac fac',round(mk_f.mean(),2),'dNLL g2 sum-share',round(n1[mk_f].sum()/n1.sum(),2),'cf rows',round(mk_c.mean(),2),'cf ach share',round(ach[mk_c].sum()/ach.sum(),2))
```