NE START atoms also carry `kq[0]`: the z error drops from 0.0015–0.002 to exactly 0.0 when it is included. So my in-range dose slope, the r = 1.0 target and the bin-16 originals all need recomputing with kq[0]. Insulin END atoms still do not reproduce, so they carry some other field.

The safe response is to constrain the gate orthogonal to every kq and categorical row. Then its value on START and END atoms depends only on code, kind and bins, whatever those extra fields are. For margins and compensation I include END variants with every optional field.

This run rebuilds everything on the corrected atoms:
- in-range slope dm, r = 1.0 target, bin-16 originals,
- full-orthogonal gate, per-unit κ, compensation means over realistic factual atoms,
- fit, bf16 error, margins and factual Δu,
- saves the edit if the checks pass.

Then a dose_edit test can tell whether ne_x2 moves toward +4.85 without disturbing the other three readouts.

```python
aNE0=aNE
def aNE(r,cap=False): return aNE0(r,cap)+KQ[0]
INSU=[aIN2(r,3,2,0,None,None,None) for r in np.linspace(0.01,17.5,120)]
NEA=[aNE(r) for r in np.linspace(0.01,1.0,60)]
xtra=[0]+[KQ[k] for k in range(4)]+[CAT[n][1] for n in CAT]+[KQ[0]+CAT['action'][1],KQ[0]+CAT['status'][1]]
ENDS=[aEND(c,u_,ru)+x for (c,u_,ru) in ((8,3,2),(9,4,3)) for x in xtra]
CONSf=CONS+list(Wt('embed.categorical.discharged_to.weight'))
def gate_f(bl,c8,k2):
    t=[c8,0.0]+[0,0,k2]+[0]*nE+[bl]*16+[0.5]+[bl]*16+[0.5]+[0]+[0]*6+[0]+[0]*(len(UN)+len(RU)); t=t+[0]*(len(CONSf)-len(t))
    w=np.linalg.lstsq(np.array(CONSf),np.array(t),rcond=None)[0]; return w
AT=np.array(INSU+NEA+ENDS); act=gel(AT@W10.T+b10); zA=AT@wZ
Mr=m1(np.array([aNE(r) for r in rr])); dm=np.polyfit(rr,Mr,1)[0]; Ap=np.array([aNE(r) for r in rp]); Mo=m1(Ap); m10=m1(aNE(1.0)[None])[0]
At16=np.array([aNE(r) for r in (1.02,1.2,1.5,1.8,2.0)]); Mt16=m10+((np.array([1.02,1.2,1.5,1.8,2.0])-1)[:,None])*dm[None]
gw=gate_f(-1.0,-4,-3); print('|gw|',np.linalg.norm(gw).round(2),'G: NE16',round(float(aNE(1.5)@gw),3),'NE.9',round(float(aNE(.9)@gw),3),'ins16',round(float(INSU[60]@gw),3),'ins1',round(float(INSU[5]@gw),3),'END max',round(float(max(e@gw for e in ENDS)),3))
Ws,fe,eb,pmx,ks=build3([(4,0),(0,-3),(2,-2),(-2,-4)],gw); print('fit',round(fe,3),'bf16',round(eb,3),'pmax',round(pmx,2),'kap',np.round(ks,2))
W0n,b0n,W2n,b2n=Ws; aEn=aEND(9,4,3)+KQ[0]
un=lambda a: uenc2([a,aEn],W0=W0n,b0=b0n,W2=W2n,b2=b2n); uo=lambda a: uenc2([a,aEn],W0=W10,b0=b10,W2=W12,b2=b12o)
print('factual NE du',round(max(np.abs(un(a)-uo(a)).max() for a in NEA[::4]),4),'ins du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in INSU[::6]),4),'END du',round(max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a],W0=W10,b0=b10,W2=W12,b2=b12o)).max() for a in ENDS),4))
if pmx<=-4.4:
    for n_,t_ in zip(['encode_action.mlp1.0.weight','encode_action.mlp1.0.bias','encode_action.mlp1.2.weight','encode_action.mlp1.2.bias'],Ws): save_edit(n_,torch.tensor(t_,dtype=torch.float32))
    np.savez('/work/E5enc.npz',*Ws); print('SAVED E5 (beta=1)', edited())
open('/work/notes.md','a').write("\n- Run 273-274: atom formulas: NE START = old formula + kq[0]; insulin START = same form, code8, unit3, rate_unit2, kq0 (z_post exact 0.0). END atoms (kind2) still unresolved -> gate made orthogonal to ALL kq/categorical rows.\n- E5 = E4 + encode_action.mlp1 units (674,411,314,524): bin16 NE START -> m(1.0)+beta*(r'-1)*dm (beta=1); /work/E5enc.npz.\n")
```