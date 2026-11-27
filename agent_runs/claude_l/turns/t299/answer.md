The output was cut at 10,000 characters because `mk_zfn` has a weight dict as a default argument, which printed every tensor. I still learned the key fact: `mkW3(Gi, Gn, base)` is my latest builder for the z-channel edit. It writes `z_post.0/2` and `z_in` from these pieces:
- `K=30` PCA components of u (`Pk`), used to make the z token a copy of u;
- presence gates (`GATE`);
- dose decoders (`DEC2`, apparently what run 272 checked);
- expected-dose models (`PH`);
- injection directions (`VD_`).

`Gi` and `Gn` are the insulin and norepinephrine gains. So Stage 1 is essentially built and needs testing.

Next I'll read /work/wm2_rollout.py to see how the offline rollout and its readout are called. I'll print the structure of `GATE`/`DEC2`/`PH`/`VD_` and parameter names only (no defaults). I'll also find which saved weight dicts differ from the original and where. The aim is to call `mkW3` and evaluate it offline next run. If the dicts or builder look inconsistent, I'll write a compact builder myself.

```python
import inspect
print(open('/work/wm2_rollout.py').read()[:4300])
def psig(fn):
    try:
        out=[]
        for p in inspect.signature(fn).parameters.values():
            dv=p.default
            if dv is p.empty: out.append(p.name)
            elif isinstance(dv,(int,float,str,bool,type(None))): out.append(f"{p.name}={dv!r}")
            else: out.append(f"{p.name}=<{type(dv).__name__}>")
        return '('+', '.join(out)+')'
    except Exception: return '(?)'
for name in ['rollout','run_batch','compare','report','tf_all','tf_run','tf_collect','true_shift','mkW3','zpost_np','mk_zfn','u_syn','rres','nextmeans','desc']:
    fn=globals().get(name); print(name, psig(fn) if fn else 'MISSING')
for dn in ['GATE','DEC2','PH','VD_','FEA','DIRS','PIH']:
    dd_=globals().get(dn)
    if isinstance(dd_,dict):
        print(dn, {k:(tuple(v.shape) if hasattr(v,'shape') else (type(v).__name__ if not isinstance(v,(int,float)) else v)) for k,v in list(dd_.items())[:8]})
for wn in ['W_cur','Wt','Wb','We','Wk','Wc','Ws0','W0z']:
    Wx=globals().get(wn)
    if not isinstance(Wx,dict): continue
    diff=[k for k in W0 if k in Wx and (tuple(Wx[k].shape)!=tuple(W0[k].shape) or not torch.equal(torch.as_tensor(Wx[k]).double(),torch.as_tensor(W0[k]).double()))]
    print(wn, 'differs:', diff[:8])
```