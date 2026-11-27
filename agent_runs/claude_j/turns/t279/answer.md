The fit blew up: the readout spread on factual rows was about 2000 and the NLL was infinite. The likely cause is the Newton tilt solver. Rows whose predictive distribution sits at an edge bin have near-zero derivative, so their tilt targets explode and the ridge chases them. Unseen groups left uncentred may add to it.

The useful number is the target itself. In the main contrast the model's own direct shift is −1.66 and the extra needed, given 0.85 persistence, is only about −1.5 mg/dL per reading (tilt ≈ −0.27).

Next run: clip the tilt targets, scale the ridge to the data, centre unseen groups by the global mean, and use a wider λ range. **Confirmation:** a sane spread (around 0.1–0.3) and finite NLL costs. If the cost at full gain is under about 0.007 per glucose reading, the head route is viable.

```python
th_raw=th_star.copy(); print('th* pct',np.round(np.percentile(th_raw,[1,5,50,95,99]),3),'clipped',round((np.abs(th_raw)>1.5).mean(),4))
th_c=np.clip(th_raw,-1.5,1.5)
gm_all=Xf.mean(0)
for lam in [10,100,1000,10000]:
    thf=np.zeros(len(fac)); thc=np.zeros(len(cf))
    for k in range(5):
        tf=ff!=k; tc=fc!=k
        gm=pd.DataFrame(Xf[tf]).groupby(gkey[tf]).mean(); G=gm.reindex(gkey).values
        G=np.where(np.isnan(G),Xf[tf].mean(0)[None,:],G); Xft=Xf-G
        A=Xc[tc].T@Xc[tc]+lam*Xft[tf].T@Xft[tf]; A+=1e-2*np.trace(A)/192*np.eye(192)
        w=np.linalg.solve(A,Xc[tc].T@th_c[tc]); thf[~tf]=Xft[~tf]@w; thc[~tc]=Xc[~tc]@w
    m1,_=mstat(LGc,thc,vG,c17c); ach=m1-m0c; fr=ach[main].mean()/cf['T'][main].mean()
    out=[]
    for g in [1.0,1/max(fr,1e-3)]:
        p=mix(LGf,g*thf,vG).mean(1); out.append(round(float((-np.log(np.maximum(p[ar,fac.ob.values],1e-12))-nll0).mean()),4))
    perm=[round(float(ach[(cf.m==mm).values].mean()/cf['T'][(cf.m==mm).values].mean()),2) for mm in [0,0.5,1.5,2,3]]
    RES[lam]=(thf,thc); print('lam',lam,'frac',round(float(fr),3),'dNLL g1/gfull',out,'sd thf',round(float(thf.std()),3),'byM',perm)
```