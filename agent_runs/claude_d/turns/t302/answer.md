Insulin bundles are solved. A first-record insulin START atom reproduces the z posterior exactly (error 0.0) when it includes `kq[0]` with unit 3 and rate_unit 2. The other 14 bundles fail only because I left out the END atom of the previous insulin record that shares their minute.

This means my gate and compensation were built on wrong insulin atoms. kq[0] has gate dot +0.47, which at κ ≈ 4.4 would erode about 2 of the −4.5 margin and let the correction leak onto bin-16 insulin atoms under insulin x2.

This run:
- confirms the NE START formula (no kq) on a first-record NE-only bundle,
- determines the insulin END atom form,
- rebuilds the gate with every kq/categorical row constrained to zero,
- recomputes compensation means and margins on realistic atoms only,
- rebuilds the correction and reports fit, bf16 error, margins and factual Δu,
- saves it if all checks pass.

```python
def zerr_atoms(rw,atoms):
    c=ctx_of(int(rw.prev_row),float((rw.minute-rw.prev_min)*60)); zt=XALL[int(rw.row),960:992]
    return float(np.abs(np.asarray(zpo(np.asarray(c).reshape(-1),e_empty,uenc2(atoms))).reshape(-1)[:32]-zt).max())
TRe=pd.read_csv('/data/train.csv'); TRe=TRe[TRe.stay_id.isin(S400)&(TRe.event=='norepinephrine')]
cn=STALL[(STALL.has_ne)&(~STALL.has_ins)]; cn=cn[[(s,m) not in measm for s,m in zip(cn.stay_id,cn.minute)]]
ok=0
for _,rw in cn.iterrows():
    g=TRe[TRe.stay_id==rw.stay_id]; S_=g[np.round(g.hour*60).astype(int)==rw.minute]; E_=g[np.round(g.end_hour*60).astype(int)==rw.minute]
    if len(S_)==1 and len(E_)==0:
        r=float(S_.rate.iloc[0]); print('NE first-rec z err: plain',round(zerr_atoms(rw,[aNE(r)]),5),'| +kq0..3',[round(zerr_atoms(rw,[aNE(r)+KQ[k]]),4) for k in range(4)]); ok+=1
    if ok>=2: break
aEi=lambda kq: aEND(8,3,2)+(KQ[kq] if kq is not None else 0)
for _,rw in cand.iloc[1:4].iterrows():
    g=TRi[TRi.stay_id==rw.stay_id]; S_=g[np.round(g.hour*60).astype(int)==rw.minute]; nE_=int((np.round(g.end_hour*60).astype(int)==rw.minute).sum()); r=float(S_.rate.iloc[0])
    print('ins START+END',nE_,[round(zerr_atoms(rw,[aIN2(r,3,2,0,None,None,None)]+[aEi(k)]*nE_),5) for k in (None,0,1,2,3)])
```