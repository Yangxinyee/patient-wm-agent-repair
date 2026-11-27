The earlier run stopped before it defined the weight shortcuts (`LW`, `W1`, `b1`, `W2`, `b2`). I'm defining them and rerunning the same register-layout search unchanged.

```python
LW = Wd['last_numeric.weight']; W1=Wd['attr_heads.value_bin.0.weight']; b1=Wd['attr_heads.value_bin.0.bias']; W2=Wd['attr_heads.value_bin.2.weight']; b2=Wd['attr_heads.value_bin.2.bias']
t0=time.time(); res=[]; la_=np.log1p(age)/10
for usebins in [0,1]:
    Bv = Wd['embed.bins.value.weight'][lb] if usebins else np.repeat(Wd['embed.bins.value.weight'][0][None,:],len(FR),0)
    for zc in range(8):
        for ac in range(8):
            if ac==zc: continue
            rest=[c for c in range(8) if c not in (zc,ac)]
            for k in [1,2]:
                for hs in itertools.combinations(rest,k):
                    R = Bv + np.outer(zl,LW[:,zc]) + np.outer(la_,LW[:,ac]) + np.outer(has,LW[:,list(hs)].sum(1))
                    rg,rm = evalR(R); res.append((rg/52+rm/20, rg, rm, usebins, zc, ac, hs))
    print('usebins',usebins,'t',round(time.time()-t0))
res.sort(key=lambda r:r[0]); print([(round(r[1],2),round(r[2],2),r[3:]) for r in res[:8]])
```