DW2 passes the factual check, but it makes every dose readout worse than the original model.

| Readout | DW2 | Original |
|---|---|---|
| Insulin x2 | −1.12 | −2.01 |
| Insulin x1/2 | +0.53 | +1.16 |
| NE x2 | +0.13 | +0.37 |
| NE x1/2 | +0.04 | −0.06 |

So the strong −7.2 of the old edit depended on its first-bundle leak. With the leak gone, the edit damps the model's own response to both drugs.

My best explanation is DW2's leftover response to ordinary slot tokens. That response (rms 312) is larger than the intended drug signal. It feeds state deviations back into the update and shrinks every effect, which fits insulin and NE shrinking together.

**This run:** build a "quiet" delta:
- zero response to the first-bundle slot tokens;
- the same response as DW2 along the main directions of the z18 (insulin) and z28 (NE) signals;
- otherwise the smallest possible response to ordinary slot, e, u and z tokens.

I save it at 2× strength so its effect stands out clearly from rollout noise. If the diagnostics show it is not actually quieter, or it loses the signal, I revert to the original weights instead.

**How the next factual and dose_edit passes will read:**
- **Confirmed:** factual stays within budget, the NE readouts return near the original, and insulin x2 becomes more negative than −2.0.
- **Channel has the wrong sign:** insulin x2 becomes less negative than −2.0. In my next run I will flip the gain.

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
DW2=np.load('/work/DW2.npy')
Msum=sum(Y.T@Y/len(Y) for Y in Ys.values()); tr=np.trace(Msum)/192
dYs={}
for c in (18,28):
    Zc=Z.copy(); Zc[:,c]+=0.1; dYs[c]=n1(Zc@Wz.T+bz)-Ys['z']
Vs=[]
for c in (18,28):
    _,s_,Vt=np.linalg.svd(dYs[c],full_matrices=False); Vs.append(Vt[:3].T); print('sv z%d'%c, np.round(s_[:4]/s_[0],3).tolist())
Cm=np.concatenate([X1]+Vs,1); R=np.concatenate([np.zeros((384,4)), DW2@Vs[0], DW2@Vs[1]],1)
def rms(D,Y): return float(np.sqrt(np.mean(np.sum((Y@D.T)**2,1))))
def rep(D): return ' '.join('%s %.1f'%(k,rms(D,Y)) for k,Y in Ys.items())+' sig18 %.1f sig28 %.1f'%(rms(D,dYs[18]),rms(D,dYs[28]))
print('DW2 ', rep(DW2))
res={}
for lamf in (1e-4,1e-3,1e-2):
    Mi_=np.linalg.inv(Msum+lamf*tr*np.eye(192)); D=R@np.linalg.solve(Cm.T@Mi_@Cm, Cm.T@Mi_); res[lamf]=D
    print('lam %g |D|/|DW2| %.2f '%(lamf,np.linalg.norm(D)/np.linalg.norm(DW2))+rep(D))
s18=rms(DW2,dYs[18]); pick=None
for lamf in (1e-3,1e-2,1e-4):
    D=res[lamf]
    if np.linalg.norm(D)/np.linalg.norm(DW2)<=3 and rms(D,dYs[18])>=0.7*s18 and all(rms(D,Y)<=1.05*rms(DW2,Y)+1 for Y in Ys.values()):
        pick=lamf; break
if pick is not None:
    np.save('/work/QD.npy',res[pick]); SE(kvn, A(load_original(kvn)).astype(np.float64)+2.0*res[pick]); print('SAVED 2xQD lam',pick, edited())
else:
    for n in list(edited()): drop_edit(n)
    print('REVERTED', edited())
open('/work/notes.md','a').write('\n- fwd12 factual DW2: +0.0014 PASS. fwd13 dose DW2: ins x2 -1.12, half +0.53, ne x2 0.13, half 0.04 -> WORSE than orig (damping; strong DW_full effect needed the leak). run297: QD = quiet delta (zero first-bundle, keep DW2 resp on top3 PCs of z18/z28 signal, min resp on tokens) saved /work/QD.npy; installed 2xQD (else reverted). 3 runs left after this.\n')
```