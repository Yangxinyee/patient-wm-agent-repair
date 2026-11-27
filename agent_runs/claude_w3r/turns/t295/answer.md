`GRa` maps each glucose-finger reading to its predicting state: `prow` is a row of SX, plus the gap, last value, age and observed value. Glucose normalisation is median 164 / IQR 52; codes are glucose_finger=6, map_arterial=7.

Now the head-side quantities:
- natural with_code for ~1700 glucose rows, via `wc_row` from the SX states;
- the with_code change per unit injection along the insulin and norepinephrine y-directions (finite differences through out_norm and the h slots);
- which `value_bin.0` units are dead;
- the minimum-variance read of each injection that is blind to the other;
- the separation of the glucose code embedding from MAP and the other codes along a register-orthogonal direction (for gating).

```python
Wh0=np.asarray(W['attr_heads.value_bin.0.weight'],float); bh0=np.asarray(W['attr_heads.value_bin.0.bias'],float)
ONw_=ONw.double().numpy(); ONb_=ONb.double().numpy()
def LNn(x,g,b): m=x.mean(-1,keepdims=True); v=((x-m)**2).mean(-1,keepdims=True); return (x-m)/np.sqrt(v+1e-5)*g+b
def wc_of(prow,gap,lv,age,dy=None):
    h=SX[prow,192:960].reshape(4,192).astype(float); o=SX[prow,:192].astype(float); z=SX[prow,960:].astype(float)
    if dy is not None: h=h+dy[None]; o=LNn(h.mean(0),ONw_,ONb_)
    return np.asarray(wc_row(o,h,gap,lv,age,z),float).ravel()
G_=GRa.reset_index(drop=True); WC=np.array([wc_of(int(r.prow),r.gap,r.lv,r.age_s) for r in G_.itertuples()])
DQ={nm:np.array([(wc_of(int(r.prow),r.gap,r.lv,r.age_s,10*TY[nm])-WC[i])/10 for i,r in enumerate(G_.iloc[:300].itertuples())]) for nm in TY}
for nm in DQ: dm=DQ[nm].mean(0); print(nm,'|dq|',round(float(np.linalg.norm(dm)),4),'consistency cos',round(float(np.mean(DQ[nm]@dm/np.linalg.norm(DQ[nm],axis=1)/np.linalg.norm(dm))),4))
preWC=WC@Wh0.T+bh0; dead=np.where(preWC.max(0)<-2)[0]; print('value_bin.0 dead units (gf rows max<-2):',len(dead),dead.tolist())
Sig=np.cov(WC.T)+1e-8*np.eye(192); Si=np.linalg.inv(Sig); RD={}
for nm,oth in [('ins','ne'),('ne','ins')]:
    Dm=np.stack([DQ[nm].mean(0),DQ[oth].mean(0)],1); M=Dm.T@Si@Dm; r=Si@Dm@np.linalg.solve(M,np.array([1.,0.])); RD[nm]=r
    print(nm,'read: natural sd per unit-injection',round(float(np.sqrt(r@Sig@r)),4),'=> noise in rate units (AY=10)',round(float(np.sqrt(r@Sig@r)/10),4))
Ec=np.asarray(W['embed.code.weight'] if 'embed.code.weight' in W else Wt['embed.code.weight'].numpy(),float)
Rsp=np.vstack([Wt['embed.bins.value.weight'].numpy(),Wt['embed.bins.rate.weight'].numpy(),Wt['embed.numeric.weight'].numpy().T]); Rsp=Rsp-Rsp.mean(0)
Pq,_=np.linalg.qr(Rsp.T); proj=lambda v: v-Pq@(Pq.T@v)
for c,nm in [(6,'gf'),(7,'map')]:
    others=[j for j in range(13) if j!=c]; gdir=proj(Ec[c]-Ec[others].mean(0)); gdir/=np.linalg.norm(gdir)
    sep=Ec[others]@gdir-Ec[c]@gdir; print(nm,'gate: min sep to other codes',round(float(-sep.max()),3),'(map/gf sep',round(float(Ec[7]@gdir-Ec[6]@gdir),3),') natural sd of g.wc on gf rows',round(float((WC@gdir).std()),4))
print('VOCAB keys',list(VOCAB.keys()))
```