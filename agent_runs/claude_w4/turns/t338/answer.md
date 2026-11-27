Run 291 mostly printed function signatures, and its diagnostics failed. The cause: my print helper overwrote a global `P`, which `LN1` (and probably `prelude_b`) use as a weight-name prefix string. The function sources cannot be retrieved, so I have to infer what the build functions do by checking the weights.

This run will:
- restore `P` by searching for the prefix that makes `LN1(u_const)` reproduce the stored `LN1c`;
- list the large arrays in the kernel and the contents of the saved `.npz` files, to find stored per-bundle tokens;
- list the prelude attention weight names;
- check how the head-1 value rows of the current edit relate to `B` (A-component replaced or added).

These decide whether I can refit the readout B on attention-weighted token mixtures at all bundles, including those before the first drug. I have 8 runs after this one, so the output is kept short.

```python
import numpy as np, torch
prefixes=set()
for n in NAMES:
    ps=n.split('.')
    for i in range(1,len(ps)):
        prefixes.add('.'.join(ps[:i])+'.'); prefixes.add('.'.join(ps[:i]))
ok=None
for pfx in sorted(prefixes, key=len):
    globals()['P']=pfx
    try:
        y=LN1(u_const[None,:].astype(np.float32))
        y=np.asarray(y.detach().numpy() if isinstance(y,torch.Tensor) else y,dtype=np.float64).reshape(-1)
        if y.shape[0]==192 and np.abs(y-LN1c).max()<1e-3: ok=pfx; break
    except Exception: pass
print('P restored:', repr(ok))
if ok is not None: P=ok
print('attn names:', [n for n in NAMES if 'prelude.0' in n][:14])
ar=sorted([(k_,tuple(v_.shape)) for k_,v_ in list(globals().items()) if not k_.startswith('_') and isinstance(v_,(np.ndarray,torch.Tensor)) and v_.ndim>=2], key=lambda t:-int(np.prod(t[1])))
print('ARR2:', ' '.join(f'{k}{s}' for k,s in ar)[:1400])
for fnm in ['rep_drug','an1','chan','v6_parts','v5_parts','dirs265','v4_parts']:
    try:
        z=np.load(f'/work/{fnm}.npz'); print(fnm, ' '.join(f'{k}{z[k].shape}' for k in z.files[:14]))
    except Exception as ex: print(fnm,'err',type(ex).__name__)
kvo=load_original('core.prelude.0.attn.kv.weight').numpy().astype(np.float64)[240:288]
kvc=load('core.prelude.0.attn.kv.weight').numpy().astype(np.float64)[240:288]
G=A.T@kvc; Go=A.T@kvo
for nm,ref in [('B^T',B.T),('M B^T',M@B.T),('Minv B^T',np.linalg.inv(M)@B.T)]:
    print(f'A^T kv vs {nm}: replace err {np.linalg.norm(G-ref)/np.linalg.norm(ref):.3g}  additive err {np.linalg.norm(G-Go-ref)/np.linalg.norm(ref):.3g}')
Pa=np.eye(48)-A@A.T
print('off-A kv change frac', np.linalg.norm(Pa@(kvc-kvo))/max(np.linalg.norm(kvc-kvo),1e-12), '| |A^T kvo| rel', np.linalg.norm(Go)/np.linalg.norm(G))
with open('/work/notes.md','a') as f:
    f.write(f'\n- run291/292: global P (weight prefix used by LN1/prelude_b) was clobbered by my print helper; restored P={ok!r}. Funcs: prelude(Xs4,e,u,z,Wkv,Wo), prelude_b(T,Wkv,Wo), chs(T,Wkv,Wout), enc(atoms,pre), build7(B,tIv,tNv,kI,kN,rbI). No source available.\n  PLAN: refit B on attention-weighted token mixtures at ALL bundles (targets 0 where no drug) to kill pre-drug leak; then build7 at s~.6.\n')
```