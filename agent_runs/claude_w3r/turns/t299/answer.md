`wc_row` returns (with_code 192, two 32-vectors, transition row index — 112 = 6·17+10, glucose code with last bin 10). `pick` was never defined because the previous run failed on its first line. Defining it and rerunning; I also keep the transition index for the compensation step.

```python
pick=lambda r: r[0].detach().double().numpy()
def wc_full(prow,gap,lv,age,dy=None):
    h=SX[prow,192:960].reshape(4,192).astype(float); o=SX[prow,:192].astype(float); z=SX[prow,960:].astype(float)
    if dy is not None: h=h+dy[None]; o=LNn(h.mean(0),ONw_,ONb_)
    return wc_row(T32(o),T32(h),gap,lv,age,T32(z))
WCr=[wc_full(int(r.prow),r.gap,r.lv,r.age_s) for r in G_.itertuples()]; WC=np.array([pick(r) for r in WCr]); TIDX=np.array([r[3] for r in WCr])
DQ={nm:np.array([(pick(wc_full(int(r.prow),r.gap,r.lv,r.age_s,10*TY[nm]))-WC[i])/10 for i,r in enumerate(G_.iloc[:300].itertuples())]) for nm in TY}
for nm in DQ: dm=DQ[nm].mean(0); print(nm,'|dq|',round(float(np.linalg.norm(dm)),4),'cos',round(float(np.mean(DQ[nm]@dm/np.linalg.norm(DQ[nm],axis=1)/np.linalg.norm(dm))),4))
preWC=WC@Wh0.T+bh0; dead=np.where(preWC.max(0)<-2)[0]; print('dead units:',len(dead),dead.tolist())
Sig=np.cov(WC.T)+1e-8*np.eye(192); Si=np.linalg.inv(Sig); RD={}
for nm,oth in [('ins','ne'),('ne','ins')]:
    Dm=np.stack([DQ[nm].mean(0),DQ[oth].mean(0)],1); r=Si@Dm@np.linalg.solve(Dm.T@Si@Dm,np.array([1.,0.])); RD[nm]=r
    print(nm,'read noise (rate units, AY=10)',round(float(np.sqrt(r@Sig@r)/10),4))
Ec=Wt['embed.code.weight'].double().numpy(); Rsp=np.vstack([Wt['embed.bins.value.weight'].numpy(),Wt['embed.bins.rate.weight'].numpy(),Wt['embed.numeric.weight'].numpy().T]); Rsp=Rsp-Rsp.mean(0)
Pq,_=np.linalg.qr(Rsp.T); GD={}
for c,nm in [(6,'gf'),(7,'map')]:
    oth=[j for j in range(13) if j!=c]; g_=Ec[c]-Ec[oth].mean(0); g_=g_-Pq@(Pq.T@g_); g_/=np.linalg.norm(g_); GD[nm]=g_
    print(nm,'gate min sep',round(float(-(Ec[oth]@g_-Ec[c]@g_).max()),3),'map-gf',round(float(Ec[7]@g_-Ec[6]@g_),3),'nat sd',round(float((WC@g_).std()),4))
print('VOCAB keys',list(VOCAB.keys()))
```