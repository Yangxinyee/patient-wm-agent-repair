The reconstruction of `build_v3` checks out:
- It edits only `core.prelude.0.attn.kv.weight` rows 240–287 (head-1 values) and `core.prelude.0.attn.out.weight` columns 48–95 (head-1 output).
- The three value directions are orthonormal.
- The output maps them exactly to −gI·dI, +gN·dN and +sP·qd. So insulin pushes the slots along −dI and NE along +dN.

The stored Jacobian arrays (`Jm`, `nuG`) give only weak alignment (cos ≈ 0.17–0.26). I don't know what perturbation that Jacobian was taken with, so I won't trust it. The real downstream effect will be measured with a forward pass.

Before spending one, I'll check two things that could make v3 harmful or useless:
1. **Constant offsets.** The original head-1 values may already have a component along aI3, aN3 or aP3. Remapping those directions would then add an extra term to every bundle, including measurement bundles, and could hurt the factual fit.
2. **Readout accuracy.** The `tI`/`fN`/`pI` readouts should recover the recorded insulin rate, NE rate and presence at the drug bundles. At the no-action token they should give about 0.

I'll also compare the size of dI and dN with the typical slot-state norm.
- **Safe to test:** the original projections on the new directions are near zero and the readouts track the rates.
- **Needs a fix first:** a large or variable original projection would mean v3 perturbs the model even without drugs.

```python
n1w=load_original('core.prelude.0.norm1.weight').numpy(); n1b=load_original('core.prelude.0.norm1.bias').numpy()
def LN1(u):
    u=np.atleast_2d(u); m=u.mean(-1,keepdims=True); v=((u-m)**2).mean(-1,keepdims=True); return (u-m)/np.sqrt(v+1e-5)*n1w+n1b
V1=kv_o[240:288]
Xu=LN1(Uk); vals=Xu@V1.T
try:
    u0=np.asarray(u_for(None,None),dtype=np.float64).ravel(); print('u_const norm %.3f'%np.linalg.norm(u0))
except Exception as e:
    u0=None; print('u_for(None,None) failed:',e)
for nm,a_ in [('aI3',aI3),('aN3',aN3),('aP3',aP3)]:
    p=vals@a_; s=f'{nm}: drug-bundle orig proj mean {p.mean():+.4f} sd {p.std():.4f} [{p.min():+.3f},{p.max():+.3f}]'
    if u0 is not None: s+=f'  u_const proj {float((LN1(u0)@V1.T)@a_):+.4f}'
    print(s)
print('RI range',RI.min(),RI.max(),'RN range',RN.min(),RN.max(), 'nI>0',(RI>0).sum(),'nN>0',(RN>0).sum())
tIr=Xu@BB3['tI']; fNr=Xu@BB3['fN']; pIr=Xu@BB3['pI']
for nm,r,t in [('tI',tIr,RI),('fN',fNr,RN)]:
    e=r-t; print(f'{nm} vs rate: rms err {np.sqrt((e**2).mean()):.4f} max|err| {np.abs(e).max():.3f} (present rows rms {np.sqrt((e[t>0]**2).mean()):.4f}, absent rows rms {np.sqrt((e[t==0]**2).mean()):.4f})')
print('pI on ins-present mean %.3f sd %.3f; absent mean %.3f sd %.3f'%(pIr[RI>0].mean(),pIr[RI>0].std(),pIr[RI==0].mean(),pIr[RI==0].std()))
if u0 is not None:
    x0=LN1(u0); print('no-action readouts: tI %.4f fN %.4f pI %.4f'%(float(x0@BB3['tI']),float(x0@BB3['fN']),float(x0@BB3['pI'])))
H=X3[:,192:960].reshape(-1,4,192); print('slot norm mean %.2f sd %.2f; out norm %.2f; |dI| %.2f |dN| %.2f'%(np.linalg.norm(H,axis=2).mean(),np.linalg.norm(H,axis=2).std(),np.linalg.norm(X3[:,:192],axis=1).mean(),np.linalg.norm(dI),np.linalg.norm(dN)))
W1o=wo_o[:,48:96]; print('typical orig head-1 output norm at drug bundles %.2f'%np.linalg.norm(vals@W1o.T,axis=1).mean())
```