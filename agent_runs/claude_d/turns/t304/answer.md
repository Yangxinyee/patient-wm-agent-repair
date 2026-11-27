The rebuilt gate is exact on every realistic atom class:
- NE START in bin 16: +1.0
- NE START in range: −2.0
- insulin START in bin 16: −3.0
- insulin START in range: −6.0
- END atoms (all variant fields): at most −3.0

The crash is a stale-length bug: `build3` still slices NE atoms as `NEA[:40]`, which no longer matches the z array built from all 60. I am redefining `build3` to use the full atom set, which also makes the margin check cover in-range NE rates up to 1.0. Then I rerun the build, the factual-Δu checks and the conditional save.

```python
def build3(spec,gw,units=(674,411,314,524),beta=1.0):
    Mt=m10+beta*(rp-1)[:,None]*dm[None]; C2=Mt-Mo
    Phi=np.array([gel(c+b*(zp_-zm)) for c,b in spec]).T; V=np.linalg.lstsq(Phi,C2,rcond=None)[0]
    GA=AT@gw; W0n=W10.copy(); b0n=b10.copy(); W2n=W12.copy(); b2n=b12o.copy(); ks=[]
    for (c,b),j,vj in zip(spec,units,V):
        k=float(np.max((c+b*(zA-zm)+4.5)/(1-GA))); ks.append(k)
        b2n+=W12[:,j]*act[:,j].mean(); W0n[j]=k*gw+b*wZ; b0n[j]=-k+c-b*zm; W2n[:,j]=vj
    eb=np.abs(m1_bf(At16,W0n,b0n,W2n,b2n)-Mt16).max(); pmx=max(float((AT@W0n[j]+b0n[j]).max()) for j in units[:len(spec)])
    return (W0n,b0n,W2n,b2n),np.abs(Phi@V-C2).max(),eb,pmx,ks
Ws,fe,eb,pmx,ks=build3([(4,0),(0,-3),(2,-2),(-2,-4)],gw); print('fit',round(fe,3),'bf16',round(eb,3),'pmax',round(pmx,2),'kap',np.round(ks,2))
W0n,b0n,W2n,b2n=Ws; aEn=aEND(9,4,3)+KQ[0]
un=lambda a: uenc2([a,aEn],W0=W0n,b0=b0n,W2=W2n,b2=b2n); uo=lambda a: uenc2([a,aEn],W0=W10,b0=b10,W2=W12,b2=b12o)
print('factual NE du',round(max(np.abs(un(a)-uo(a)).max() for a in NEA[::4]),4),'ins du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in INSU[::6]),4),'END du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in ENDS),4))
print('bin16 u err vs target',[round(float(np.abs(un(aNE(r))-u_from_m(m10+(r-1)*dm)).max()),3) for r in (1.02,1.5,2.0)])
if pmx<=-4.4:
    for n_,t_ in zip(['encode_action.mlp1.0.weight','encode_action.mlp1.0.bias','encode_action.mlp1.2.weight','encode_action.mlp1.2.bias'],Ws): save_edit(n_,torch.tensor(t_,dtype=torch.float32))
    np.savez('/work/E5enc.npz',*Ws); print('SAVED E5 (beta=1)', edited())
open('/work/notes.md','a').write("\n- Run 273-276: atom formulas: NE START = old formula + kq[0]; insulin START: code8, unit3, rate_unit2, kq0 (z_post exact). END atoms unresolved -> gate orthogonal to ALL kq/categorical rows (gate_f(-1,-4,-3)).\n- E5 = E4 + encode_action.mlp1 units (674,411,314,524): bin16 NE START -> m(1.0)+beta*(r'-1)*dm (beta=1); /work/E5enc.npz.\n")
```