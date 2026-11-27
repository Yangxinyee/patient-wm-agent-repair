last_numeric has rank 3 on setA. But both decoders have large norms (373 for insulin, 64 for NE). Even after removing last_numeric, their outputs vary by 30-47 (insulin) and 7-9 (NE) within each code. That still mixes between-cell variation (last-bin embeddings, which the compensation removes) with real within-cell noise.

So I will measure the within-cell spread on setA after removing last_numeric, using the decoders projected orthogonal to its 3-dimensional subspace. If the spread drops to near the train-replica value (1.3 / 0.28), the decoders transfer once constrained to that subspace's complement. If it stays large, the q replica itself is off and I must use low-norm, heavily regularised decoders.

```python
def within_sd(s_,c_,lb_,codes):
    out_=[]
    for c in codes:
        r_=[]; 
        for b in np.unique(lb_[c_==c]):
            mk=(c_==c)&(lb_==b)
            if mk.sum()>5: r_.append(s_[mk]-s_[mk].mean())
        out_.append(round(np.concatenate(r_).std(),3))
    return out_
P3=Vt[:3]
for nm,w0,codes in [('ins',wI,[0,1]),('ne',wN,[2,3])]:
    wp=w0-P3.T@(P3@w0)
    print(nm,'raw wc',within_sd(wc_@w0,codeA,lbA,codes),'wc-lnum',within_sd((wc_-lnum)@w0,codeA,lbA,codes),'proj',within_sd(wc_@wp,codeA,lbA,codes),
          'train Xf proj',within_sd(Xf@wp,cf,lbf,codes),'resp ins2/ne2',resp(wp,keysI[0],codes)[0],resp(wp,keysN[0],codes)[0])
```