`wc_row` returns a tuple of 4 items, not a single vector. I will print its structure once and take the 192-element with_code vector, then run the same diagnostics as planned.

```python
r_test=wc_row(torch.tensor(SX[int(G_.prow[0]),:192],dtype=torch.float64),torch.tensor(SX[int(G_.prow[0]),192:960].reshape(4,192),dtype=torch.float64),G_.gap[0],G_.lv[0],G_.age_s[0],torch.tensor(SX[int(G_.prow[0]),960:],dtype=torch.float64))
print('wc_row returns',[ (type(t).__name__, tuple(t.shape) if hasattr(t,'shape') else t) for t in r_test])
def pick(r):
    for t in (r if isinstance(r,(tuple,list)) else [r]):
        a=np.asarray(t.detach().double().numpy() if torch.is_tensor(t) else t,float).ravel()
        if a.size==192: return a
def wc_of(prow,gap,lv,age,dy=None):
    h=SX[prow,192:960].reshape(4,192).astype(float); o=SX[prow,:192].astype(float); z=SX[prow,960:].astype(float)
    if dy is not None: h=h+dy[None]; o=LNn(h.mean(0),ONw_,ONb_)
    T_=lambda a: torch.tensor(a,dtype=torch.float64); return pick(wc_row(T_(o),T_(h),gap,lv,age,T_(z)))
WC=np.array([wc_of(int(r.prow),r.gap,r.lv,r.age_s) for r in G_.itertuples()])
DQ={nm:np.array([(wc_of(int(r.prow),r.gap,r.lv,r.age_s,10*TY[nm])-WC[i])/10 for i,r in enumerate(G_.iloc[:300].itertuples())]) for nm in TY}
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