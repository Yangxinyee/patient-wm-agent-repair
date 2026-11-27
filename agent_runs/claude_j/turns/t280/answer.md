With clipped targets the fit behaves, and the insulin head-only route looks cheap. At λ=100 the out-of-fold readout reaches 49% of the needed extra shift. Scaled to 100%, it costs only +0.0006 NLL per glucose reading, about 0.0003 overall. The achieved/needed ratio is similar for every dose multiplier (0.38–0.51), so the response is roughly linear in the dose change.

One puzzle remains. The original model's teacher-forced direct shift (−1.66) is only about 1.7× smaller than its rollout shift (−2.75). With persistence 0.85 I would expect about 4×. So the feedback through sampled readings in rollouts may be weaker than assumed, and that sets how much direct effect I need.

Next run:
- **Profiles:** compare, hour by hour, the tool's rollout shift from fwd_1, the teacher-forced direct shift in M, and the true shift S, for both drugs.
- **Drug B readout:** run the same readout fit on the MAP rows, with persistence 0.85 and 0.6.

If the rollout shift is flat rather than ramping, the effective persistence is low and I will need a larger gain, calibrated with the tool.

```python
de=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print('de',de.shape,list(de.columns)[:10])
ecol=[c for c in de.columns if 'edit' in c.lower()]; hcol=[c for c in de.columns if 'hour' in c.lower()]; scol=[c for c in de.columns if 'shift' in c.lower()]
if ecol and hcol and scol:
    pv=de.groupby([ecol[0],de[hcol[0]].round().astype(int)])[scol[0]].mean().unstack()
    for e,r in pv.iterrows(): print('tool',e,np.round(r.values[:12],2).tolist())
for drug,ev in [('insulin','glucose_finger'),('norepinephrine','map_arterial')]:
    q_=Mi[(Mi.drug==drug)&(Mi.h0==12)&(Mi.m==2.0)&(Mi.event==ev)]
    g_=q_.groupby(q_.hour.round().astype(int))
    print(drug,'dir',np.round((g_.mean_cf.mean()-g_.mean_f.mean()).values[:12],2).tolist())
    print(drug,'S  ',np.round(g_.S.mean().values[:12],2).tolist(),'hours',int(g_.ngroups))
def build(drug,EV):
    fa=Mi[(Mi.drug=='none')&Mi.event.isin(EV)].sort_values(['stay','event','hour']).copy()
    fa['pobs']=fa.groupby(['stay','event'])['obs'].shift(1)
    fa['lb']=[0 if pd.isna(p) else int(fb_(e,p)) for e,p in zip(fa.event,fa.pobs)]
    fa['ob']=[int(fb_(e,o)) for e,o in zip(fa.event,fa.obs)]
    c=Mi[(Mi.drug==drug)&Mi.event.isin(EV)&(Mi.m!=1.0)].copy()
    fk=dict(zip(zip(fa.stay,fa.event,fa.b),fa.ri)); c['fi']=[fk.get(k,-1) for k in zip(c.stay,c.event,c.b)]; c=c[c.fi>=0]
    hs=c.groupby(['stay','h0','m','hour'])['S'].mean().rename('Sh').reset_index().sort_values(['stay','h0','m','hour'])
    hs['Sp']=hs.groupby(['stay','h0','m'])['Sh'].shift(1).fillna(0.0)
    return fa,c.merge(hs[['stay','h0','m','hour','Sp']],on=['stay','h0','m','hour'],how='left')
def fitro(fa,c,v,rho,lams,clip=1.5,tag=''):
    c17c=np.stack([cenof(e) for e in c.event])[:,None,:]; LGc=LG[c.ri.values]; LGf=LG[fa.ri.values]
    T=(c.S-rho*c.Sp-(c.mean_cf-c.mean_f)).values; th=np.clip(tsolve(LGc,v,c17c,T),-clip,clip)
    Xc=WC[c.ri.values]-WC[c.fi.values]; Xf=WC[fa.ri.values]; mn=((c.h0==12)&(c.m==2.0)).values
    st=np.array(sorted(fa.stay.unique())); fo={s:i%5 for i,s in enumerate(st)}
    ff=fa.stay.map(fo).values; fc=c.stay.map(fo).values; gk=(fa.event+'_'+fa.lb.astype(str)).values
    ar=np.arange(len(fa)); ob=fa.ob.values; n0=-np.log(mix(LGf,np.zeros(len(fa)),v).mean(1)[ar,ob]); m0,_=mstat(LGc,np.zeros(len(c)),v,c17c)
    print(tag,'rho',rho,'T main',round(float(T[mn].mean()),2),'dir main',round(float((c.mean_cf-c.mean_f)[mn].mean()),2))
    for lam in lams:
        thf=np.zeros(len(fa)); thc=np.zeros(len(c))
        for k in range(5):
            tf=ff!=k; tc=fc!=k; gm=pd.DataFrame(Xf[tf]).groupby(gk[tf]).mean(); G=gm.reindex(gk).values
            G=np.where(np.isnan(G),Xf[tf].mean(0)[None,:],G); Xft=Xf-G
            A=Xc[tc].T@Xc[tc]+lam*Xft[tf].T@Xft[tf]; A+=1e-2*np.trace(A)/192*np.eye(192)
            w=np.linalg.solve(A,Xc[tc].T@th[tc]); thf[~tf]=Xft[~tf]@w; thc[~tc]=Xc[~tc]@w
        m1,_=mstat(LGc,thc,v,c17c); ach=m1-m0; fr=ach[mn].mean()/T[mn].mean(); out=[]
        for g in [1.0,1/max(fr,1e-3)]:
            p=mix(LGf,g*thf,v).mean(1); out.append(round(float((-np.log(np.maximum(p[ar,ob],1e-12))-n0).mean()),4))
        perm=[round(float(ach[(c.m==mm).values].mean()/T[(c.m==mm).values].mean()),2) for mm in [0,0.5,1.5,2,3]]
        print(' lam',lam,'frac',round(float(fr),3),'dNLL g1/gfull',out,'byM',perm)
MAPE=['map_arterial','map_cuff']; faN,cN=build('norepinephrine',MAPE)
for rho in [0.85,0.6]: fitro(faN,cN,vM,rho,[10,100,1000],tag='NE')
```