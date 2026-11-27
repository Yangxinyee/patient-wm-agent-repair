Gating now works. Decoder output is exactly 0 at every u-absent local bundle. The gate read is clean (positive residual 0.026, max 0.08 on negatives), and the no-action garbage of the rate reads dropped to ≤2.2, so side effects are small.

Decoder accuracy is still mediocre. Insulin has residuals of 0.26–0.72 and biases up to +0.5 at low rates, which would distort the ×½ and low-rate ×2 differences. I test whether that is a capacity problem or an information problem. I compare 8, 12 and 40 random GELU units, on the drug's own two reads versus all four rate reads. If more units on four reads reach a residual of about 0.15, I will use about 10 units per drug from the roughly 20 repurposable ones.

```python
def rf_fit(cols,y,K,lam=40.,seed=0):
    rng=np.random.default_rng(seed); R=Vu2[:,cols]; m=y>0
    C=rng.normal(size=(len(cols),K))/R[m].std(0)[:,None]/np.sqrt(len(cols)/2); b=-(R[m].mean(0)@C)+rng.normal(size=K)*1.2+0.5
    H=Fnn.gelu(torch.tensor(lam*(Vu2[:,[0]]-1)+R@C+b)).numpy(); o=np.linalg.lstsq(H,y,rcond=None)[0]; pr=H@o
    return C,b,o,lam,pr
def summ(pr,y): return [(round(float(np.std(pr[s_]-y[s_])),2),round(float(np.mean(pr[s_]-y[s_])),2)) for lo,hi in [(0,1e-9),(1e-9,.3),(.3,1),(1,2),(2,5),(5,30)] for s_ in [((y>=lo)&(y<hi)) if hi>1e-9 else (y==0)] if s_.sum()>5]
for nm,own,y in [('ins',[1,2],yi),('ne',[3,4],yn)]:
    for cols in [own,[1,2,3,4]]:
        for K in [8,12,40]:
            best=min([(float(np.std(rf_fit(cols,y,K,seed=s)[4]-y)),s) for s in range(8)])
            pr=rf_fit(cols,y,K,seed=best[1])[4]; print(nm,cols,K,'seed',best[1],'res',round(best[0],3),summ(pr,y))
```