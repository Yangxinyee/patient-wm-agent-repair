Run 282 showed that the key tools are still in the kernel:
- `mkW3(Gi, Gn, base)`, the builder for the extra-token edit, with Gi and Gn the insulin and norepinephrine gains;
- `evalW(W, stays, ...)`, the offline rollout evaluator, which by default runs the four dose edits;
- `tfnll(W, stays)`, which computes the teacher-forced NLL;
- `true_shift` and the IRF arrays `IRF_I` (12 lags) and `IRF_N` (6 lags), which give the effect sizes I estimated from the records.

Their source code is gone, so I can only learn their inputs and outputs by calling them. The output was cut before the non-array variables and the /work file list, so I still don't know the name of the weight dict to pass as `base`.

Next I will find the weight-dict candidates and the batch stay list. Then I will run one trial: `mkW3` at gains (1, 1), which tensors it changes, and its offline readouts and teacher-forced NLL on 40 batch-1 stays, timed. This tells me whether the builder works and roughly how its gains scale the effects. One result would support the extra-token plan: readouts that move from the original's values (about −3.3 for insulin ×2) toward my estimated effects (about −14), with NLL change within budget. A failure, NaNs, or a big NLL jump would mean it needs repair.

```python
import json, time, types, torch, numpy as np
g = dict(globals()); lines=[]
for k,v in sorted(g.items()):
    if k.startswith('_') or isinstance(v,(types.FunctionType,types.ModuleType,np.ndarray,type)): continue
    if isinstance(v, dict) and len(v)>0:
        k0=list(v)[0]; lines.append(f"D {k}[{len(v)}]:{type(v[k0]).__name__} {str(k0)[:30]}")
    elif isinstance(v,(int,float,bool,np.floating)): lines.append(f"S {k}={v:.4g}" if isinstance(v,float) else f"S {k}={v}")
    elif isinstance(v,(list,tuple)): lines.append(f"L {k}[{len(v)}]")
print(' | '.join(lines)[:3500])
bj=json.load(open('/work/batches.json')); print('batches:', {k:len(v) for k,v in bj.items()} if isinstance(bj,dict) else len(bj))
cands=[k for k,v in g.items() if isinstance(v,dict) and len(v)>=170 and all(isinstance(x,torch.Tensor) for x in list(v.values())[:3])]
zo=load_original('z_post.0.weight') if 'z_post.0.weight' in NAMES else None
for k in cands:
    v=g[k]; zh=v.get('z_head.weight'); 
    print(k, len(v), 'z_head zero:', bool(zh is not None and float(zh.abs().max())==0),
          'z_post* differs from orig:', [n for n in v if n.startswith('z_') and n in NAMES and not torch.equal(v[n].float(), load_original(n))][:6])
print('IRF_I', np.round(IRF_I,2), 'IRF_N', np.round(IRF_N,2))
stays=list(BUN.keys())[:40]
base=[g[k] for k in cands if bool(float(g[k]['z_head.weight'].abs().max())==0)]
base=base[0] if base else g[cands[0]]
t=time.time(); Wt=mkW3(1.0,1.0,base); print('mkW3 %.1fs'%(time.time()-t), type(Wt).__name__)
print('changed vs base:', [(n, round(float((Wt[n].float()-base[n].float()).norm()),3)) for n in Wt if n in base and not torch.equal(Wt[n],base[n])])
t=time.time(); r=evalW(Wt,stays); print('evalW %.1fs'%(time.time()-t)); print(repr(r)[:1500])
t=time.time(); r0=tfnll(base,stays); r1=tfnll(Wt,stays); print('tfnll %.1fs'%(time.time()-t), repr(r0)[:200], '||', repr(r1)[:200])
```