Found the leak. At a stay's first bundle the state is still zero, so the slot tokens are just `core.slots`. Head 1 puts 24% of its attention on them, and the readout B returns a fake drug signal of about [126, 102, 20] there; a typical insulin rate is 1.8.

At every other non-drug bundle, before or after the first drug, head 1 attends entirely to the constant drug token, and the readout is exactly 0. So the pre-drug damage is a single huge kick at the first bundle, which the state carries until the first drug bundle resets it.

The ridge refit removes the kick, but the readout's accuracy at drug bundles gets clearly worse (error 0.15 vs 0.11). A more surgical fix is to project B off the four first-bundle slot vectors and, optionally, the leading principal directions of event tokens. That makes the readout exactly zero there and leaves the drug-bundle readout nearly unchanged.

Plan for this run:
1. Compare projection variants on drug-bundle accuracy and on the leak at synthetic first bundles, with and without an event token.
2. Keep the best variant and rebuild the edit with `build7` at gains 2.91 / 3.45, which predict a post-drug factual cost of about 0.
3. Check that the edit was saved with the intended structure.

A factual pass should then show the pre-drug ΔNLL close to 0. If it does not, the leak comes from somewhere else, such as the event token.

```python
import numpy as np, torch, hashlib
S=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
f64=lambda t:(t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)).astype(np.float64)
Ls=LNa(slots); Le=LNa(Td[:,4]); Ue,se,_=np.linalg.svd((Le-Le.mean(0)).T,full_matrices=False)
Bt0=B+Bo
def proj_B(k_e):
    V=np.column_stack([Ls.T, Le.mean(0)[:,None]]+([Ue[:,:k_e]] if k_e>0 else []))
    Q,_=np.linalg.qr(V); return (Bt0-Q@(Q.T@Bt0))-Bo
rng=np.random.default_rng(1); nf=300
Tf2=np.concatenate([np.repeat(slots[None],nf,0), Td[rng.integers(0,len(Td),nf),4][:,None], np.repeat(u_const[None,None],nf,0), np.repeat(zinB[None,None],nf,0)],1)
Xf2,wf2=mixes(Tf2)
say('first-bundle(with e proxy) h1 att: slots %.3f e %.3f uconst %.3f z %.3f'%tuple(np.r_[wf2[:,:,:4].sum(-1).mean(), wf2[:,:,4].mean(), wf2[:,:,5].mean(), wf2[:,:,6].mean()]))
r3=lambda x: np.round(np.sqrt((x**2).mean(tuple(range(x.ndim-1)))),3)
say('curB: drugErr',r3(Xd@Bt0-F[:,None]),'first',r3(Xf@Bt0),'first+e',r3(Xf2@Bt0),'e tok',r3(Le@Bt0))
res={}
for k_e in (0,5,10,20,40):
    Bp=proj_B(k_e); Bt=Bp+Bo; res[k_e]=Bp
    say(f'proj ke{k_e}: drugErr',r3(Xd@Bt-F[:,None]),'meas',r3(Xm@Bt),'first',r3(Xf@Bt),'first+e',r3(Xf2@Bt),'e tok',r3(Le@Bt))
base=np.sqrt(((Xd@Bt0-F[:,None])**2).mean((0,1)))
ok=[k for k in res if np.all(np.sqrt(((Xd@(res[k]+Bo)-F[:,None])**2).mean((0,1)))<=base*1.15+0.003)]
kbest=max(ok) if ok else 0
Bsel=res[kbest]; say('chosen k_e', kbest)
np.savez('/work/v10_parts.npz', B=Bsel, kI=2.91, kN=3.45, k_e=kbest)
h0=hashlib.md5(f64(load('core.prelude.0.attn.kv.weight')).tobytes()).hexdigest()[:8]
r=build7(Bsel, tI5, tN5, 2.91, 3.45)
kvN=f64(load('core.prelude.0.attn.kv.weight'))
h1=hashlib.md5(kvN.tobytes()).hexdigest()[:8]
say('build7 ret', type(r).__name__, len(r) if isinstance(r,(tuple,list)) else '', '| kv hash', h0, '->', h1, '| edited', edited())
say('kv struct err %.2g'%(np.linalg.norm(A.T@kvN[240:288]-A.T@kvO[240:288]-M@Bsel.T)/np.linalg.norm(M@Bsel.T)))
dWb=f64(load('core.b_in.weight'))-WbO; dWo=f64(load('core.prelude.0.attn.out.weight'))-WoO
say('|dWb| %.3g |dWo| %.3g'%(np.linalg.norm(dWb),np.linalg.norm(dWo)))
with open('/work/notes.md','a') as f:
    f.write('\n- run294 KEY: leak origin = FIRST bundle of each stay: slot tokens=core.slots (h=0), head1 attends 24% to slots, B readout there = [126,102,20] (!). All other non-drug bundles: head1 attends 100% to u_const, readout 0.\n- run295: v10 = B projected orthogonal to LN(core.slots) (+e-token mean/PCs), build7(Bsel,tI5,tN5,2.91,3.45) (s_I=.6,s_N=.5). B in /work/v10_parts.npz\n'+'\n'.join('  '+s for s in S)+'\n')
```