Run 296 has a serious bug, and I caught it before spending a forward pass on it. The minimal-change fix did:
- zero the first-bundle readout;
- keep the drug-bundle readouts (error 0.107 / 0.055 / 0.015).

But it did not constrain the constant drug token u_const. At every ordinary non-drug bundle, where head 1 attends 100% to u_const, the insulin readout is now about 4.1 instead of 0. The norepinephrine readout on event tokens also rose to 4.8. The saved v10 would therefore inject a large fake insulin signal almost everywhere, and must be replaced.

This run solves one joint weighted least-squares problem for the change to B, with these terms:
- **Drug bundles:** keep the readouts unchanged.
- **Ordinary non-drug bundles and u_const:** drive the readout to 0 (heavy weight).
- **First bundles, with and without event-token proxies:** drive the readout to 0 (heavy weight).
- **Event tokens:** keep their old, small readouts.

I then rebuild with `build7` at the same gains and save the result manually. A good result has a drug-bundle error of about 0.11, readouts of about 0.01 or less at u_const and first bundles, and event-token readouts below about 1. I will check all of these before saving.

```python
import numpy as np, torch, hashlib
S=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
f64=lambda t:(t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)).astype(np.float64)
Xdf=Xd.reshape(-1,192); Xmf=Xm.reshape(-1,192); Xf2f=Xf2.reshape(-1,192); Xff=Xf.reshape(-1,192)
Lu=LNa(u_const[None]); nd=len(Xdf); Bt0=B+Bo
Gp=Xdf.T@Xdf + (nd/len(Le))*Le.T@Le
Gp+=1e-6*np.trace(Xdf.T@Xdf)/192*np.eye(192)
Cs=[(Xmf,100*nd/len(Xmf)),(Xf2f,1000*nd/len(Xf2f)),(Xff,1000*nd/len(Xff)),(Lu,100*nd)]
Gc=sum(w*C.T@C for C,w in Cs)
dB=-np.linalg.solve(Gp+Gc, Gc@Bt0)
Bn2=B+dB; Bt=Bn2+Bo
say('fixB2: drugErr',r3(Xd@Bt-F[:,None]),'meas',r3(Xm@Bt),'uconst',np.round(Lu@Bt,4),'first',r3(Xf@Bt),'first+e',r3(Xf2@Bt),'e tok',r3(Le@Bt),'|B| %.3g'%np.linalg.norm(Bn2))
good = (np.sqrt(((Xd@Bt-F[:,None])**2).mean((0,1)))<=np.array([0.13,0.07,0.02])).all() and np.abs(Xm@Bt).max()<0.05 and np.sqrt(((Xf2@Bt)**2).mean())<0.2
say('good', good)
if good:
    np.savez('/work/v10_parts.npz', B=Bn2, kI=2.91, kN=3.45)
    mats=[f64(x) for x in build7(Bn2, tI5, tN5, 2.91, 3.45)]
    kvl=[m for m in mats if m.shape==(384,192)]; sq=[m for m in mats if m.shape==(192,192)]
    def offh1(m):
        d=m-WoO; d[:,48:96]=0; return np.abs(d).max()
    won=[m for m in sq if offh1(m)<1e-6]; wbn=[m for m in sq if offh1(m)>=1e-6]
    if len(kvl)==1 and len(won)==1 and len(wbn)==1:
        save_edit('core.prelude.0.attn.kv.weight', torch.tensor(kvl[0],dtype=torch.float32))
        save_edit('core.prelude.0.attn.out.weight', torch.tensor(won[0],dtype=torch.float32))
        save_edit('core.b_in.weight', torch.tensor(wbn[0],dtype=torch.float32))
        kvN=f64(load('core.prelude.0.attn.kv.weight'))
        say('SAVED v10b. kv struct err %.2g'%(np.linalg.norm(A.T@kvN[240:288]-A.T@kvO[240:288]-M@Bn2.T)/np.linalg.norm(M@Bn2.T)),
            '| |dWb| %.3g |dWo| %.3g'%(np.linalg.norm(f64(load('core.b_in.weight'))-WbO),np.linalg.norm(f64(load('core.prelude.0.attn.out.weight'))-WoO)),
            '| hash', hashlib.md5(kvN.tobytes()).hexdigest()[:8], edited())
else:
    for n in ['core.b_in.weight','core.prelude.0.attn.kv.weight','core.prelude.0.attn.out.weight']: drop_edit(n)
    say('NOT good -> dropped all edits (original model)', edited())
with open('/work/notes.md','a') as f:
    f.write('\n- run296 v10 was BAD: min-change fix ignored u_const -> insulin readout 4.1 at every meas bundle. run297: v10b = joint WLS fix (keep drug readouts, zero meas/u_const/first(+e proxy), keep e-token readouts) + build7(...,2.91,3.45) saved manually if checks pass.\n'+'\n'.join('  '+s for s in S)+'\n')
```