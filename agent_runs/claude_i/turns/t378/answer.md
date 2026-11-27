The flipped setting (insulin gain −1, NE gain −0.75) also weakens every dose response. Insulin x2 is −1.86 against −2.01, x1/2 +0.93 against +1.16, NE x2 +0.22 against +0.37, and NE x1/2 −0.056 against −0.064. So this channel damps the dose response whichever sign it has; the effect is mainly quadratic in the gain. No setting of it can help, and I am abandoning it. Every factual-passing edit I have tested so far scores worse on the readouts than the original model.

With two runs left, I'll try one different, low-leak edit. The plan is to amplify the model's own dose pathway where the dose actually enters, in the u token:
- The value projection's response to within-class variation of the u token is multiplied by about 2.5 for insulin, 4 for NE and 3 for bundles with both drugs. Within a class, that variation comes only from dose and rate.
- The response to the class means, to the empty-drug token and to the first-bundle slot tokens is held at zero.
- Everything else is kept as quiet as possible, using the same solve as before.
- All other edits are dropped, so the model is the original plus this one change.
- If the diagnostics show leaks into slot, e or z tokens, the run reverts to the original weights instead.

The next factual and dose_edit passes will judge it:
- **Confirmed:** factual within about +0.003, and all four dose readouts larger in size than the original's.
- **Refuted:** either check fails. Then the last run reverts to the original model.

```python
import numpy as np, torch
def A(x):
    try: return x.detach().cpu().numpy()
    except AttributeError: return np.asarray(x)
def SE(n,x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float32))
    try: save_edit(n,x)
    except Exception: save_edit(n,torch.from_numpy(x))
kvn='core.prelude.0.attn.kv.weight'
W0=A(load_original(kvn)).astype(np.float64); WV=W0[192:]
def rms(D,Y): return float(np.sqrt(np.mean(np.sum((Y@D.T)**2,1))))
ok=False
try:
    F=np.load('/work/fwd_3/states_original.npz'); ks=list(F.keys()); print('keys', ks)
    ki=[k for k in ks if 'ins' in k.lower()][0]; kn=[k for k in ks if 'nor' in k.lower() or k.lower().endswith('_ne')][0]
    hi=np.asarray(F[ki]).astype(bool).ravel(); hn=np.asarray(F[kn]).astype(bool).ravel(); U=Ys['u']
    assert len(hi)==len(U)==len(hn)
    cls={'I':hi&~hn,'N':hn&~hi,'B':hi&hn,'O':~hi&~hn}; print({k:int(v.sum()) for k,v in cls.items()})
    u0=U[cls['O']].mean(0); print('u0 spread %.4f'%np.sqrt(((U[cls['O']]-u0)**2).sum(1).mean()))
    gam={'I':2.5,'N':4.0,'B':3.0}; cons=[X1,u0[:,None]]; req=[np.zeros((192,X1.shape[1])),np.zeros((192,1))]; PCs={}
    for c in ('I','N','B'):
        mk=cls[c]
        if mk.sum()<20: continue
        mu=U[mk].mean(0); _,s_,Vt=np.linalg.svd(U[mk]-mu,full_matrices=False)
        k=int(min(4,np.sum(s_>=0.1*s_[0]))); V=Vt[:k].T; PCs[c]=V
        print(c,'k',k,'sv',np.round(s_[:5]/s_[0],2).tolist())
        cons+=[mu[:,None],V]; req+=[np.zeros((192,1)),(gam[c]-1)*WV@V]
    Cu=np.concatenate(cons,1); Ru=np.concatenate(req,1)
    Mi2=np.linalg.inv(Msum+1e-3*tr*np.eye(192)); Gm=Cu.T@Mi2@Cu
    DV=Ru@np.linalg.solve(Gm+1e-9*np.trace(Gm)*np.eye(len(Gm)), Cu.T@Mi2)
    Du=np.zeros((384,192)); Du[192:]=DV; W0V=np.zeros_like(W0); W0V[192:]=WV
    for k_,Y in Ys.items(): print(k_,'rms Du %.3f W_V %.3f'%(rms(Du,Y),rms(W0V,Y)))
    for c in PCs:
        Cc=U[cls[c]]-U[cls[c]].mean(0); print(c,'amp %.2f'%(rms(Du,Cc)/max(rms(W0V,Cc),1e-9)))
    print('zero chk %.2e %.2e'%(np.abs(DV@X1).max(), np.abs(DV@u0).max()))
    ok=all(rms(Du,Ys[k_])<=0.03*rms(W0V,Ys[k_]) for k_ in ('slot','e','z'))
except Exception as ex: print('ERR', repr(ex)[:200])
for n in list(edited()): drop_edit(n)
if ok:
    SE(kvn, W0+Du); np.save('/work/Du.npy',Du); print('INSTALLED u-amp', edited())
else: print('REVERTED to original', edited())
open('/work/notes.md','a').write('\n- fwd16 factual QD(kI=-1,kN=-.75) +0.0007; fwd17 dose -1.86,+0.93,0.22,-0.056 -> QD channel DAMPS for both signs (quadratic). QD abandoned; all factual-passing QD/DW2 variants worse than original on readouts.\n- run299: u-amplification Du (V-path response to within-class u-token variation x gamma I2.5 N4 B3; zero on class means, empty u, first-bundle slots; quiet solve). Installed ALONE (all else dropped) if quiet, else reverted. Run 300 = final keep/revert ONLY.\n')
```