The dose_edit pass on the projected edit (full strength) adds almost nothing, so scaling it down is pointless.

| Readout | Projected edit | Original |
|---|---|---|
| Insulin x2 | −2.71 | −2.01 |
| Insulin x1/2 | +3.54 | +1.16 |
| NE x2 | −0.07 | +0.37 |
| NE x1/2 | −0.03 | −0.06 |

The older unprojected edit gave −7.2 / +5.65 at a factual cost of only +0.037.

My explanation: the old kv delta was designed to give almost no response to later-bundle slot tokens, which contain the slot vectors plus the state. Projecting out the first-bundle slot directions broke that. It created a new leak at every bundle, which matches the broad factual damage in fwd_10.

The right fix is a minimum-disturbance correction:
- **Constraint:** zero response to the 4 first-bundle slot tokens.
- **Objective:** change the response to ordinary tokens (later slot tokens, e, u and z tokens) as little as possible, weighted by their covariance in the saved states.

This run builds that correction and prints the response, per token type, of the old delta, the projected delta and the new delta. It also prints the strength of the intended z18/z28 signal. If the new delta stays quiet on ordinary tokens (no more than 2× the old response), it is saved. Otherwise I revert to the original weights.

The next factual and dose_edit passes will test it:
- **Confirmation:** factual within about +0.005, with insulin shifts near the old −7 / +5.6.
- **Refutation:** factual still fails, or the shifts collapse. That would mean the strong effect depended on the leak itself.

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
DW=np.load('/work/DW_kv_full.npy').astype(np.float64); P=np.load('/work/edit_parts.npz'); DWp=np.load('/work/DWp.npz')['DWp']
w1=A(load_original('core.prelude.0.norm1.weight')).astype(np.float64); b1=A(load_original('core.prelude.0.norm1.bias')).astype(np.float64)
def n1(x): x=np.asarray(x,np.float64); m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+1e-5)*w1+b1
S=A(load_original('core.slots')).astype(np.float64); X1=n1(S).T
print('shapes', {k:np.shape(globals()[k]) for k in ('HHt','Ee','Uu','ZZt') if k in globals()})
fa=[n for n in NAMES if 'flow' in n]; print('flow', [(n,tuple(SHAPES[n])) for n in fa])
Ys={}
try:
    H=np.asarray(HHt,np.float64).reshape(-1,4,192); a=None
    for n in fa:
        if int(np.prod(SHAPES[n]))%192==0: a=A(load_original(n)).astype(np.float64).reshape(-1,192); break
    L=[]
    for dt in (300.,3300.):
        f=np.exp(-dt*np.exp(a)) if a is not None else 1.0
        L.append(n1(H*f+S).reshape(-1,192))
    Ys['slot']=np.concatenate(L,0)
except Exception as ex: print('ERRs', repr(ex)[:150])
for k,g in (('e','Ee'),('u','Uu')):
    try: Ys[k]=n1(np.asarray(globals()[g],np.float64).reshape(-1,192))
    except Exception as ex: print('ERR'+k, repr(ex)[:100])
try:
    Z=np.asarray(ZZt,np.float64).reshape(-1,32); Wz=P['Wzo'].astype(np.float64); bz=P['bz'].astype(np.float64)
    Ys['z']=n1(Z@Wz.T+bz)
except Exception as ex: print('ERRz', repr(ex)[:100])
M=sum(Y.T@Y/len(Y) for Y in Ys.values()); lam=1e-4*np.trace(M)/192; Mi=np.linalg.inv(M+lam*np.eye(192))
DW2=DW-DW@X1@np.linalg.solve(X1.T@Mi@X1, X1.T@Mi)
def rms(D,Y): return float(np.sqrt(np.mean(np.sum((Y@D.T)**2,1))))
rat=[]
for k,Y in Ys.items():
    r0,rp,r2=rms(DW,Y),rms(DWp,Y),rms(DW2,Y); rat.append(r2/max(r0,1e-6)); print(k, len(Y), 'rms DW %.1f DWp %.1f DW2 %.1f'%(r0,rp,r2))
print('leak DW2', np.round(np.linalg.norm(DW2@X1,axis=0),3).tolist(), '|DW2|/|DW| %.3f'%(np.linalg.norm(DW2)/np.linalg.norm(DW)))
if 'z' in Ys:
    for c in (18,28):
        Zc=Z.copy(); Zc[:,c]+=0.1; dY=n1(Zc@Wz.T+bz)-Ys['z']
        print('sig z%d DW %.1f DWp %.1f DW2 %.1f'%(c,rms(DW,dY),rms(DWp,dY),rms(DW2,dY)))
np.save('/work/DW2.npy', DW2)
if len(Ys)>=3 and max(rat)<2.0:
    SE(kvn, A(load_original(kvn)).astype(np.float64)+DW2); print('SAVED DW2', edited())
else:
    for n in list(edited()): drop_edit(n)
    print('REVERTED', edited())
open('/work/notes.md','a').write('\n- fwd10 factual P12(proj DW_full): +0.107 FAIL broad. fwd11 dose P12: ins x2 -2.71, half +3.54, ne x2 -0.07, half -0.03 (weak). Projection broke quietness on later tokens.\n- run296: DW2 = min-disturbance fix (zero first-bundle slot response, M=later token cov) saved /work/DW2.npy; saved if quiet (ratio<2) else ALL REVERTED. Next: factual+dose_edit.\n')
```