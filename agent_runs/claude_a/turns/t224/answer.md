Two clear findings:
- Decoding the insulin rate directly from `n3_0` or from E is poor: residual sd 0.37–0.83 units/h, with slopes of 0.61–1.13 across dose versions. Head 3's staging stays necessary.
- E (the prelude output) gives a clean insulin gate, with lowest insulin score 0.33 against highest other score −0.22 and no overlaps. It also gives a clean any-drug gate.

That suggests a 5-dim design where the gate lives in the prelude FFN, before E:
1. A raw staging dim reads head 3 through the prelude out-projection. I scale it down so that even its huge off-target values stay bounded, about 30 at most, which keeps the prelude layer norms intact. The prelude norm excludes this dim from E.
2. Two prelude FFN units apply the insulin gate and write the gated rate into E at the memory index jI. E's index space is separate from h's, so the same index can serve both roles.
3. `b_in` copies (E_jI − E_jR) into the memory dim of h.

So insulin needs only jWI plus jI, NE needs jWN plus jN, and the reference jR is shared.

This run checks the prerequisites:
- separability of the insulin and NE gates on the prelude FFN input, LN3 of the post-attention residual;
- the spread of that layer norm's sd across drug bundles, which sets the multiplicative noise on each write;
- whether a tiny off-target penalty lowers the off/on ratio of the head-3 decoder much while keeping accuracy.

How I will read it:
- Clean gates, a sd CV around 10% or less, and an off/on ratio that allows a bounded raw staging confirm the design.
- If not, I will need a different gate location.

```python
NP={}; SD={}
for k in DS:
    n3s=[]; sds=[]
    for i in range(0, len(STS[k]['hf']), 2000):
        s=slice(i,i+2000); pr=prelude(P2, STS[k]['hf'][s], DS[k]['e'][s], DS[k]['u'][s], STS[k]['z'][s])
        n3s.append(pr['n3'][:,:4,:][:,:,keep]); sds.append(pr['X1'][:,:4,:].std(-1))
    NP[k]=torch.cat(n3s).numpy(); SD[k]=torch.cat(sds).numpy()
for gname,lab in (('ins',lambda k: RS[k]>0),('ne',lambda k: isNE)):
    Xg=np.concatenate([NP[k].reshape(-1,len(keep)) for k in DS]); yg=np.concatenate([np.repeat(np.where(lab(k),1.,-1.),4) for k in DS])
    wg_=ridge(Xg,yg,1e-1); s_=ap(Xg,wg_); pos=s_[yg>0]; neg=s_[yg<0]
    print(f"prelude-n3 gate {gname}: pos min {pos.min():.3f} neg max {neg.max():.3f} | #neg>posmin {int((neg>pos.min()).sum())} #pos<negmax {int((pos<neg.max()).sum())}")
sdd=np.concatenate([SD[k][(RS[k]>0)|isNE].ravel() for k in DS]); print('X1 sd at drug bundles: mean', round(sdd.mean(),2), 'CV', round(sdd.std()/sdd.mean(),3), 'p1/p99', np.round(np.percentile(sdd,[1,99]),2))
onmax=yd.max()
for woff in [0, 1e-6, 1e-5, 1e-4]:
    w_=np.linalg.solve(Xd1.T@Xd1+woff*Xo1.T@Xo1+1e-6*np.eye(49), Xd1.T@yd)
    rd_=Xd1@w_-yd; od=np.abs(Xo1@w_)
    print('woff',woff,'| drug resid sd',round(rd_.std(),3),'| off max',round(od.max(),1),'ratio off/onmax',round(od.max()/onmax,1))
```