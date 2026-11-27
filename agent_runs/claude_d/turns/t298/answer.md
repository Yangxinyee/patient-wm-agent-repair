The bf16 problem has a clear structure. Insulin z-scores reach 10 under insulin x2. So any correction unit whose pre-activation rises with z needs a large gate gain κ to stay silent on insulin, and large κ is what makes bf16 errors big (0.5 here, against a natural 0.008). Two of the three designs also leaked onto non-target atoms (pre-activation +8.4 and +0.4).

Fix: use only constant and falling ramps in z (β ≤ 0). For those, large insulin z pushes the unit further off, and only the small lower z range (≥ −1.5) needs margin. Each unit also gets its own minimal κ, computed from the actual atom set with a −4.5 safety margin.

This run searches a few gate structures and bases for the lowest bf16 error. It builds the best one, checks factual Δu and bin-16 target error, and saves it if non-target pre-activations stay ≤ −4.5.

```python
zA=np.array(INSU+NEA[:40]+ENDS)@wZ
def build3(spec,gw,units=(674,411,314,524),beta=1.0):
    Mt=m10+beta*(rp-1)[:,None]*dm[None]; C2=Mt-Mo
    Phi=np.array([gel(c+b*(zp_-zm)) for c,b in spec]).T; V=np.linalg.lstsq(Phi,C2,rcond=None)[0]
    GA=np.array(INSU+NEA[:40]+ENDS)@gw
    W0n=W10.copy(); b0n=b10.copy(); W2n=W12.copy(); b2n=b12o.copy(); ks=[]
    for (c,b),j,vj in zip(spec,units,V):
        k=float(np.max((c+b*(zA-zm)+4.5)/(1-GA))); ks.append(k)
        b2n+=W12[:,j]*act[:,j].mean(); W0n[j]=k*gw+b*wZ; b0n[j]=-k+c-b*zm; W2n[:,j]=vj
    eb=np.abs(m1_bf(At16,W0n,b0n,W2n,b2n)-Mt16).max()
    pmx=max(float((np.array(INSU+NEA[:40]+ENDS)@W0n[j]+b0n[j]).max()) for j in units[:len(spec)])
    return (W0n,b0n,W2n,b2n),np.abs(Phi@V-C2).max(),eb,pmx,ks
best=None
for gp in [(-1.5,-8,-4),(-2.5,-10,-6),(-1.0,-4,-3)]:
    gw=gate(*gp)
    for spec in [[(4,0),(0,-3),(2,-2)],[(3,0),(1,-2),(-1,-3)],[(4,0),(0,-3),(2,-2),(-2,-4)]]:
        Ws,fe,eb,pmx,ks=build3(spec,gw)
        print(gp,spec,'fit',round(fe,3),'bf16',round(eb,3),'pmax',round(pmx,2),'kap',np.round(ks,2))
        if pmx<=-4.4 and (best is None or max(fe,eb)<best[0]): best=(max(fe,eb),gp,spec,Ws)
print('CHOSEN',best[:3]); W0n,b0n,W2n,b2n=best[3]
un=lambda a: uenc2([a,aE],W0=W0n,b0=b0n,W2=W2n,b2=b2n); uo=lambda a: uenc2([a,aE],W0=W10,b0=b10,W2=W12,b2=b12o)
print('factual NE du',round(max(np.abs(un(a)-uo(a)).max() for a in NEA[:40:3]),4),'ins/END du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in INSU[::5]+ENDS),4))
print('bin16 u err vs target', [round(float(np.abs(un(aNE(r))-u_from_m(m10+(r-1)*dm)).max()),3) for r in (1.02,1.5,2.0)], 'u move r=1->2', round(float(np.linalg.norm(u_from_m(m10+dm)-u_from_m(m10))),2))
```