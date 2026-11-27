With start-atom targets the picture changes:
- Full u encodes the insulin rate perfectly on factual data (R² 1.0).
- Head 3's 48-dim value decodes it well, pooled over the dose versions:

| Dose version | Residual sd | Slope | Rate range |
|---|---|---|---|
| factual | 0.13 | 1.05 | up to 8 |
| ×0.5 | 0.15 | 0.97 | up to 4 |
| random | 0.36 | 0.95 | up to 14 |
| ×2 | 0.48 | 0.90 | up to 16 |

- At NE-only bundles the decoder gives under 0.26 in absolute value (99th percentile).

So staging from head 3 is viable. The insulin gate on `n3_0` still overlaps (insulin min −0.03 against non-insulin max 0.44).

**Simpler option to test first.** Write the memory linearly through `b_in` from a staging dim, with no gate. That avoids a second layer norm, two FFN units and a gate fit. It works only if the decoder can also be held near zero at non-drug bundles, where head 3 attends other tokens. My old trade-off numbers used the wrong targets, so I will redo the fit:
- error at drug bundles, minimized on the pooled dose versions;
- plus a penalty on its output at non-drug slots.

I will report drug-bundle accuracy and the off-target 99th percentile and maximum for several penalty weights.

How I will read it:
- Off-target p99 below about 0.3 with drug residual sd below about 0.3 means the gateless design works.
- Otherwise I keep the gated block-0 write and look for a drug vs non-drug gate.

```python
def o3_slots(P, ds, ST_):
    out=[]
    for i in range(0, len(ST_['hf']), 2000):
        s=slice(i,i+2000); pr=prelude(P, ST_['hf'][s], ds['e'][s], ds['u'][s], ST_['z'][s])
        out.append(pr['o'][:,3,:4,:])
    return torch.cat(out).numpy()
STS={k:lib3.internals(P2,d_) for k,d_ in DS.items()}
O3={k:o3_slots(P2,DS[k],STS[k]) for k in DS}
drug={k:(RS[k]>0)|isNE for k in DS}
Xd=np.concatenate([O3[k][drug[k]].reshape(-1,48) for k in DS]); yd=np.concatenate([np.repeat(RS[k][drug[k]],4) for k in DS])
Xo=np.concatenate([O3[k][~drug[k]].reshape(-1,48) for k in DS])
Xd1=np.c_[Xd,np.ones(len(Xd))]; Xo1=np.c_[Xo,np.ones(len(Xo))]
for woff in [0, 1e-3, 1e-2, 1e-1, 1.0]:
    w_=np.linalg.solve(Xd1.T@Xd1+woff*Xo1.T@Xo1+1e-6*np.eye(49), Xd1.T@yd)
    rd_=Xd1@w_-yd; od=np.abs(Xo1@w_)
    print('woff',woff,'| drug resid sd',round(rd_.std(),3),'max',round(np.abs(rd_).max(),2),'| off p99',round(np.percentile(od,99),3),'max',round(od.max(),2))
```