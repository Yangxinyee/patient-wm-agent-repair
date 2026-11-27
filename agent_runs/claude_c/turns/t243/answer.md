For insulin the bin-16 group is not anomalous. The model gets about 0.14 of the kernel truth in both rate bands, so insulin's trained row 16 works normally. NE's use of row 16 is the suspicious one, but even in-range NE responses are slightly wrong-signed (confounding).

I want a clean test of whether the out-of-range bin drives NE x2 negative. I will set row 16 equal to row 12 (the NE cap bin) in both the rate and dose tables, using three row-mixing rank-1 ops per table, with nothing else applied:

1. Zero row 16.
2. Add row 12 to rows 12 and 16 together.
3. Halve row 12 back.

The current 50 ops (numeric x5 + head0 x2.5) are easy to rebuild, so I will clear them. I will check numerically that the composition really yields row16 = row12, using a random test matrix.

```python
def mix_test():
    W=np.random.randn(17,5); W0=W.copy()
    def r1(W,v,a): v=v/np.linalg.norm(v); return W+a*np.outer(v,v@W)
    e=np.eye(17)
    W=r1(W,e[16],-1); W=r1(W,e[12]+e[16],2); W=r1(W,e[12],-0.5)
    return np.abs(W[16]-W0[12]).max(), np.abs(W[12]-W0[12]).max(), np.abs(np.delete(W,[16],0)-np.delete(W0,[16],0)).max()
print(mix_test())
clear_ops()
e=np.eye(17)
for nm in ['embed.bins.dose.weight','embed.bins.rate.weight']:
    rank1(nm,e[16],-1.0,side='out'); rank1(nm,e[12]+e[16],2.0,side='out'); rank1(nm,e[12],-0.5,side='out')
print(len(ops()), ops()[:2])
```