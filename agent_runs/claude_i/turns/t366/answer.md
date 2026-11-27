fwd_9 shows the current C1 edit is broken. Value NLL is 1.989 against 1.533 for the original (+0.455), and it is worse on all four outputs. It must not be submitted.

The earlier full-strength edit is a better starting point. It cost only +0.037 in factual fit, and gave insulin x2 −7.2 and x1/2 +5.65. Its damage was traced to one leak: the kv delta responds to the first-bundle slot tokens.

**Plan for this run:**
* Restore that configuration: the z_post and z_in tensors from `edit_parts.npz` plus the full kv delta.
* Remove the leak by projecting the kv delta orthogonal to the normalised first-bundle slot tokens.
* Print how much of the intended signal survives, and save an insulin/NE split of the delta so later runs can rescale the two channels separately.

**What would confirm or refute this:**
* If that leak was the main cost, the next factual pass should land within about +0.005 of the original, and the dose_edit should still show about −7 for insulin x2.
* If the factual cost stays large, the leak is elsewhere. I will then scale the edit down or go back to the original weights.

```python
import numpy as np, torch
def A(x):
    try: return x.detach().cpu().numpy()
    except AttributeError: return np.asarray(x)
def SE(n,x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float32))
    try: save_edit(n,x)
    except Exception: save_edit(n,torch.from_numpy(x))
nd=lambda a,b: round(float(np.linalg.norm(np.asarray(a,np.float64)-np.asarray(b,np.float64))),3)
kvn='core.prelude.0.attn.kv.weight'
P=np.load('/work/edit_parts.npz'); C=np.load('/work/C1.npz'); DW=np.load('/work/DW_kv_full.npy').astype(np.float64)
cur={n:A(load(n)) for n in edited()}
print('cur-vs-parts', nd(cur['z_post.0.weight'],P['W1']), nd(cur['z_post.0.bias'],P['b1']), nd(cur['z_post.2.weight'],P['W2']), nd(cur['z_post.2.bias'],P['b2']), '| zin-Wzo', nd(cur['z_in.weight'],P['Wzo']), 'zin-Wz2', nd(cur['z_in.weight'],C['Wz2']), '| Wzo-orig', nd(P['Wzo'],A(load_original('z_in.weight'))), 'bz-orig', nd(P['bz'],A(load_original('z_in.bias'))))
print('DW |K|,|V|', round(float(np.linalg.norm(DW[:192])),1), round(float(np.linalg.norm(DW[192:])),1))
pre=[n for n in NAMES if 'prelude' in n and ('norm' in n or '.ln' in n or 'ln_' in n)]; print('prenorms', [(n,tuple(SHAPES[n])) for n in pre])
sl=[n for n in NAMES if 'slot' in n]; print('slots', [(n,tuple(SHAPES[n])) for n in sl])
ok=False
try:
    S=A(load_original([n for n in sl if int(np.prod(SHAPES[n]))%192==0 and int(np.prod(SHAPES[n]))<=192*8][0])).reshape(-1,192).astype(np.float64)
    def ln(x,w,b): m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+1e-5)*w+b
    vecs=[]; wl=[n for n in pre if n.endswith('weight') and tuple(SHAPES[n])==(192,)]
    for wn in wl:
        bn=wn[:-6]+'bias'; w=A(load_original(wn)).astype(np.float64); b=A(load_original(bn)).astype(np.float64) if bn in NAMES else np.zeros(192)
        X=ln(S,w,b); vecs.append(X)
        print(wn, 'leak', np.round(np.linalg.norm(DW@X.T,axis=0)).tolist(), '|DWb|', round(float(np.linalg.norm(DW@b)),1))
    Q,_=np.linalg.qr(np.concatenate(vecs,0).T)
    DWp=DW-(DW@Q)@Q.T
    print('k', Q.shape[1], '|DWp|/|DW| %.3f'%(np.linalg.norm(DWp)/np.linalg.norm(DW)), 'leak after %.4f'%max(np.abs(DWp@X.T).max() for X in vecs))
    w0=A(load_original(wl[0])).astype(np.float64)
    for c in (18,28):
        g=w0*(P['Wzo'][:,c]-P['Wzo'][:,c].mean())
        print('signal z%d keep %.3f'%(c, np.linalg.norm(DWp@g)/max(np.linalg.norm(DW@g),1e-9)))
    M=np.stack([P['vI'],P['vN']],1).astype(np.float64); Cf=np.linalg.pinv(M)@DWp[192:]; R=DWp[192:]-M@Cf
    print('decomp resid frac %.3f'%(np.linalg.norm(R)/max(np.linalg.norm(DWp[192:]),1e-9)))
    np.savez('/work/DWp.npz', DWp=DWp, Cf=Cf, R=R, Q=Q)
    ok=True
except Exception as ex: print('ERR', repr(ex)[:300])
if ok:
    SE(kvn, A(load_original(kvn)).astype(np.float64)+DWp)
    SE('z_post.0.weight',P['W1']); SE('z_post.0.bias',P['b1']); SE('z_post.2.weight',P['W2']); SE('z_post.2.bias',P['b2'])
    SE('z_in.weight',P['Wzo']); SE('z_in.bias',P['bz'])
    print('SAVED', edited())
open('/work/notes.md','a').write('\n- fwd9 factual of C1 edit: NLL 1.988 vs 1.533 orig (+0.455) -> C1 BROKEN everywhere. run295: back to edit_parts z_post/z_in + DW_full projected orthogonal to first-bundle slot tokens; saved /work/DWp.npz (DWp, Cf=coef rows on vI,vN, R resid, Q). Next: factual+dose_edit both on sel1.\n')
```