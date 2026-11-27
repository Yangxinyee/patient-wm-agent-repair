Stage 1 numbers: decode noise is 0.014 for the gate, 0.18 for the insulin r-read and 0.07–0.09 for the others. The side effect on E is small: mean |ΔE| ≈0.04 against an E norm of about 13 per slot, so the ungated injection barely disturbs the original computation. Part of the r-read noise is probably the per-token LayerNorm scale multiplying large no-action garbage, which the gate will cut off.

Stage 2 now: per drug, six gated GELU units with gate weight λ. They are fitted by least squares on UX (real tokens under all four dose edits) to output the rate. I check:
- accuracy by rate range,
- that outputs at no-action bundles in the local set are about 0 (the gate works),
- how the stage-1 decode noise propagates at action bundles.

```python
print('intercepts RC',np.round(RC,2))
Vh=N3h@RHO; ab=np.repeat(act,4)
print('decode noise action-only', np.round((Vh-Vflat)[ab].std(0),4), ' no-action V max|.|', np.round(np.abs(Vflat[~ab]).max(0),1))
Vu=N1X@RW
def build_dec(cols,y,K=6,lam=40.,seed=1):
    rng=np.random.default_rng(seed); R=Vu[:,cols]; m=y>0
    C=rng.normal(size=(len(cols),K))/R[m].std(0)[:,None]; b=-(R[m].mean(0)@C)+rng.normal(size=K)*1.0+1.0
    H=Fnn.gelu(torch.tensor(lam*(Vu[:,[0]]-1)+R@C+b)).numpy(); o=np.linalg.lstsq(H,y,rcond=None)[0]; return C,b,o,lam,H@o
DECS={}
for nm,cols,y in [('ins',[1,2],yi),('ne',[3,4],yn)]:
    best=None
    for seed in range(6):
        C,b,o,lam,pr=build_dec(cols,y,seed=seed); r_=np.std(pr-y)
        if best is None or r_<best[0]: best=(r_,seed,C,b,o,lam,pr)
    r_,seed,C,b,o,lam,pr=best; DECS[nm]=(cols,C,b,o,lam)
    rs=[round(float(np.std(pr[(y>=lo)&(y<hi)]-y[(y>=lo)&(y<hi)])),2) for lo,hi in [(0,1e-9),(1e-9,.3),(.3,1),(1,2),(2,5),(5,30)] if ((y>=lo)&(y<hi)).sum()>5]
    bs=[round(float(np.mean(pr[(y>=lo)&(y<hi)]-y[(y>=lo)&(y<hi)])),2) for lo,hi in [(0,1e-9),(1e-9,.3),(.3,1),(1,2),(2,5),(5,30)] if ((y>=lo)&(y<hi)).sum()>5]
    # local bundles: true-V vs decoded-V outputs
    Hl=Fnn.gelu(torch.tensor(lam*(Vh[:,[0]]-1)+Vh[:,cols]@C+b)).numpy()@o; Ht=Fnn.gelu(torch.tensor(lam*(Vflat[:,[0]]-1)+Vflat[:,cols]@C+b)).numpy()@o
    print(nm,'seed',seed,'res by range',rs,'bias',bs,'| local no-action out max',round(float(np.abs(Hl[~ab]).max()),4),'action noise sd',round(float((Hl-Ht)[ab].std()),3),'|o|max',round(float(np.abs(o).max()),2))
```