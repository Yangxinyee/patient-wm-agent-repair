The run failed because `wc_row` expects torch tensors (it multiplies h by a torch time-flow factor). Rerunning the same computation with torch float64 inputs; nothing else changes.

```python
def wc_of(prow,gap,lv,age,dy=None):
    h=SX[prow,192:960].reshape(4,192).astype(float); o=SX[prow,:192].astype(float); z=SX[prow,960:].astype(float)
    if dy is not None: h=h+dy[None]; o=LNn(h.mean(0),ONw_,ONb_)
    T_=lambda a: torch.tensor(a,dtype=torch.float64)
    try: r=wc_row(T_(o),T_(h),gap,lv,age,T_(z))
    except Exception: r=wc_row(T_(o).float(),T_(h).float(),gap,lv,age,T_(z).float())
    return np.asarray(r.detach().double().numpy() if torch.is_tensor(r) else r,float).ravel()
G_=GRa.reset_index(drop=True); WC=np.array([wc_of(int(r.prow),r.gap,r.lv,r.age_s) for r in G_.itertuples()])
DQ={nm:np.array([(wc_of(int(r.prow),r.gap,r.lv,r.age_s,10*TY[nm])-WC[i])/10 for i,r in enumerate(G_.iloc[:300].itertuples())]) for nm in TY}
for nm in DQ: dm=DQ[nm].mean(0); print(nm,'|dq|',round(float(np.linalg.norm(dm)),4),'consistency cos',round(float(np.mean(DQ[nm]@dm/np.linalg.norm(DQ[nm],axis=1)/np.linalg.norm(dm))),4))
preWC=WC@Wh0.T+bh0; dead=np.where(preWC.max(0)<-2)[0]; print('value_bin.0 dead units (gf rows max<-2):',len(dead),dead.tolist())
Sig=np.cov(WC.T)+1e-8*np.eye(192); Si=np.linalg.inv(Sig); RD={}
for nm,oth in [('ins','ne'),('ne','ins')]:
    Dm=np.stack([DQ[nm].mean(0),DQ[oth].mean(0)],1); M=Dm.T@Si@Dm; r=Si@Dm@np.linalg.solve(M,np.array([1.,0.])); RD[nm]=r
    print(nm,'read noise per unit-injection',round(float(np.sqrt(r@Sig@r)),4),'=> rate units (AY=10)',round(float(np.sqrt(r@Sig@r)/10),4))
Ec=Wt['embed.code.weight'].double().numpy()
Rsp=np.vstack([Wt['embed.bins.value.weight'].numpy(),Wt['embed.bins.rate.weight'].numpy(),Wt['embed.numeric.weight'].numpy().T]); Rsp=Rsp-Rsp.mean(0)
Pq,_=np.linalg.qr(Rsp.T); proj=lambda v: v-Pq@(Pq.T@v); GD={}
for c,nm in [(6,'gf'),(7,'map')]:
    others=[j for j in range(13) if j!=c]; gdir=proj(Ec[c]-Ec[others].mean(0)); gdir/=np.linalg.norm(gdir); GD[nm]=gdir
    sep=Ec[others]@gdir-Ec[c]@gdir; print(nm,'gate: min sep',round(float(-sep.max()),3),'(map-gf',round(float(Ec[7]@gdir-Ec[6]@gdir),3),') natural sd g.wc gf rows',round(float((WC@gdir).std()),4))
print('VOCAB keys',list(VOCAB.keys()))
```