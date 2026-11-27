Run 295 found two problems:

- **Projecting B failed badly.** B is a large-norm solution whose components cancel each other. Removing even the four first-bundle slot directions breaks the cancellation, and readouts everywhere jump to about 100.
- **`build7` did not save anything.** It returned a 3-tuple of weight tensors, and the kv hash did not change. The model is therefore still v9, and the `kv struct err` and `|dWb|` lines describe v9, not the new build.

Two numbers point to a better fix. At a real first bundle head 1 puts only 19% of its attention on the event token. The current event-token readouts are small (rms 0.3–0.8), so their leak there is negligible. The damage comes almost entirely from the fixed first-bundle mixture: slots, z0 and the average event token.

This run applies a minimal-change correction to B. It zeroes the readout on that fixed mixture (8 constraint vectors) while changing the drug-bundle readouts as little as possible, measured on the drug-bundle data. This should keep the drug-bundle error near 0.11, unlike the ridge refit's 0.15.

I then call `build7` at insulin and norepinephrine gains 2.91 and 3.45 and save its three returned tensors myself. I tell them apart by shape and by which head-1 columns differ from the original out-projection. Success means the first-bundle readouts are about 0, the drug-bundle error stays small, and the saved kv has the intended structure.

```python
import numpy as np, torch, hashlib
S=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
f64=lambda t:(t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)).astype(np.float64)
Xdf=Xd.reshape(-1,192); Bt0=B+Bo
Cf=np.concatenate([Xf2.mean(0), Xf[0]],0)
G=Xdf.T@Xdf; mu=1e-6*np.trace(G)/192; Gi=np.linalg.inv(G+mu*np.eye(192))
K=Cf@Gi@Cf.T
dB=-Gi@Cf.T@(np.linalg.pinv(K,rcond=1e-12)@(Cf@Bt0))
Bn=B+dB; Bt=Bn+Bo
say('fixB: drugErr',r3(Xd@Bt-F[:,None]),'meas',r3(Xm@Bt),'first',r3(Xf@Bt),'first+e',r3(Xf2@Bt),'e tok',r3(Le@Bt),'|B| %.3g->%.3g'%(np.linalg.norm(B),np.linalg.norm(Bn)))
np.savez('/work/v10_parts.npz', B=Bn, kI=2.91, kN=3.45)
r=build7(Bn, tI5, tN5, 2.91, 3.45)
mats=[f64(x) for x in r]
kvl=[m for m in mats if m.shape==(384,192)]; sq=[m for m in mats if m.shape==(192,192)]
def offh1(m):
    d=m-WoO; d[:,48:96]=0; return np.abs(d).max()
won=[m for m in sq if offh1(m)<1e-6]; wbn=[m for m in sq if offh1(m)>=1e-6]
say('tuple', [m.shape for m in mats], 'n_kv',len(kvl),'n_out',len(won),'n_bin',len(wbn))
if len(kvl)==1 and len(won)==1 and len(wbn)==1:
    save_edit('core.prelude.0.attn.kv.weight', torch.tensor(kvl[0],dtype=torch.float32))
    save_edit('core.prelude.0.attn.out.weight', torch.tensor(won[0],dtype=torch.float32))
    save_edit('core.b_in.weight', torch.tensor(wbn[0],dtype=torch.float32))
    kvN=f64(load('core.prelude.0.attn.kv.weight'))
    say('SAVED v10. kv struct err %.2g'%(np.linalg.norm(A.T@kvN[240:288]-A.T@kvO[240:288]-M@Bn.T)/np.linalg.norm(M@Bn.T)),
        '| off-A kv change %.2g'%np.abs((np.eye(48)-A@A.T)@(kvN[240:288]-kvO[240:288])).max(),
        '| rows outside 240:288 changed %.2g'%np.abs(np.delete(kvN-kvO,np.s_[240:288],0)).max(),
        '| |dWb| %.3g |dWo| %.3g'%(np.linalg.norm(f64(load('core.b_in.weight'))-WbO),np.linalg.norm(f64(load('core.prelude.0.attn.out.weight'))-WoO)),
        '| hash', hashlib.md5(kvN.tobytes()).hexdigest()[:8], edited())
with open('/work/notes.md','a') as f:
    f.write('\n- run295: projecting B fails (B large-norm w/ cancellations -> errors ~100). build7 RETURNS (kv,out,b_in) tuple, does NOT save.\n- run296: v10 = min-change fix of B (zero readout on first-bundle mixtures, preserve drug readouts in drug-data metric) + build7(Bn,tI5,tN5,2.91,3.45) saved manually. Bn in /work/v10_parts.npz.\n'+'\n'.join('  '+s for s in S)+'\n')
```