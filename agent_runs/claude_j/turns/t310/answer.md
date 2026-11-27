On bundles without a drug, head 2 attends mostly to the measurement token (0.70) and partly to the empty action token (0.25). So the rate probe w that head 2 will read must give:
- zero on the empty action token, which is the same fixed vector every time;
- close to zero on measurement tokens.

`run_cf` returns the replica's out (nb×192), h (nb×4×192) and an index, which lets me track an injected signal through the core.

Next I'll find the kernel functions that turn replica states into head outputs, giving the query q and predicted means at readings. I'll also build constrained rate probes for both drugs. The constraint: read the START rate from the action token, be exactly zero on the empty action token, and be penalised on measurement and slot tokens.

The constrained probes should keep R² ≥ 0.85 with near-zero output on non-drug tokens. If they lose most of their accuracy, I'll need the numeric-pathway encoder edit instead.

```python
cands=[n for n,v in list(globals().items()) if callable(v) and not n.startswith('_') and any(t in n.lower() for t in ('head','wc','logit','reading','pred','unit','apply','tilt'))]
print([f"{n}{inspect.signature(globals()[n])}"[:70] for n in cands][:20])
# constrained probes
e0,u0=bvd([]) if True else None
LUe=ln1(T64(u0)).numpy()
OT=np.concatenate([o for o in OTH],0)  # slot+e+z LN1 tokens at drug bundles
idx=np.random.RandomState(0).choice(len(OT),6000,replace=False); OT=OT[idx]
def cprobe(d,lam=3.0,mu=1e-3):
    Xd=np.stack([x[4] for x in R[d] if x[2] in (0.,0.5,1.,1.5,2.)]); y=np.array([x[3] for x in R[d] if x[2] in (0.,0.5,1.,1.5,2.)])
    A=Xd.T@Xd+lam*len(y)/len(OT)*OT.T@OT+1e4*np.outer(LUe,LUe)+mu*len(y)*np.eye(192)
    w=np.linalg.solve(A,Xd.T@y); p=Xd@w
    return w,1-((p-y)**2).sum()/((y-y.mean())**2).sum(), float(np.abs(OT@w).std()), float(LUe@w)
PRB={}
for d in R:
    w,r2,sdo,ue=cprobe(d); PRB[d]=w; print(d,'R2',round(r2,3),'sd_other',round(sdo,3),'empty',round(ue,4))
```