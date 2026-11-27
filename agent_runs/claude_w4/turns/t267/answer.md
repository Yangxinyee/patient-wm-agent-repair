The kernel already holds the pieces for a CPU chain on the 2319 drug bundles of my 100 tuning stays:

- prelude input tokens (`TOK`, 7 tokens per bundle),
- time‑decayed slots (`htf`),
- a mapping to state rows (`d_idx`),
- target new slots and `out` (`yt`, `ot`) from the exact GPU states.

There are also gradient‑like vectors (`g_glucose_finger`, etc.) of unknown provenance, which I won't trust.

Plan:
1. Write a fresh, transparent prelude replica that lets me add an injection `dX` to the slot residual right after attention, exactly where the head‑1 carrier lands.
2. Feed its E into `step_full`.
3. Check the resulting y and `out` against `yt`/`ot`.

If R² is about 0.99 or better, the chain prelude → step → value‑head replica is trustworthy enough to compute each candidate injection's immediate effect on predicted glucose and MAP. I can then design the injection direction on CPU and spend forward passes only on verification. A poor fit means I go empirical.

```python
import torch
pn=[n for n in NAMES if n.startswith('core.prelude')]; print(pn)
PW={n:np.asarray(load_original(n),dtype=np.float64) for n in pn}
def prelude_np(TOKx,W=PW,dX=None):
    x=np.asarray(TOKx,dtype=np.float64).copy(); N=x.shape[0]
    h1=LNn(x,W['core.prelude.0.norm1.weight'],W['core.prelude.0.norm1.bias'])
    q=(h1@W['core.prelude.0.attn.q.weight'].T).reshape(N,7,4,48).transpose(0,2,1,3)
    kv=h1@W['core.prelude.0.attn.kv.weight'].T
    k=kv[...,:192].reshape(N,7,4,48).transpose(0,2,1,3); v=kv[...,192:].reshape(N,7,4,48).transpose(0,2,1,3)
    s=q@k.transpose(0,1,3,2)/np.sqrt(48); s=np.exp(s-s.max(-1,keepdims=True)); s/=s.sum(-1,keepdims=True)
    o=(s@v).transpose(0,2,1,3).reshape(N,7,192)@W['core.prelude.0.attn.out.weight'].T
    x=x+o
    if dX is not None: x[:,:4]+=dX[:,None,:] if dX.ndim==2 else dX
    h2=LNn(x,W['core.prelude.0.norm2.weight'],W['core.prelude.0.norm2.bias'])
    x=x+G(h2@W['core.prelude.0.ffn.0.weight'].T+W['core.prelude.0.ffn.0.bias'])@W['core.prelude.0.ffn.2.weight'].T+W['core.prelude.0.ffn.2.bias']
    return LNn(x[:,:4],W['core.prelude_norm.weight'],W['core.prelude_norm.bias']), s
Ep,att_=prelude_np(TOK)
print('E vs E_ref_d rel err %.4f'%(np.linalg.norm(Ep-E_ref_d)/np.linalg.norm(E_ref_d)), ' head1 slot->u att mean %.4f'%att_[:,1,:4,5].mean())
try:
    r=step_full(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),torch.tensor(np.asarray(htf),dtype=torch.float32))
    print(type(r), [getattr(t,'shape',None) for t in (r if isinstance(r,(tuple,list)) else [r])])
    yv=np.asarray(r[0]).reshape(2319,4,192); ov=np.asarray(r[1]).reshape(2319,192)
    def r2(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()
    print('R2 y %.4f out %.4f'%(r2(yv,np.asarray(yt)),r2(ov,np.asarray(ot))), ' ot vs states out rel', np.linalg.norm(np.asarray(ot)-xo[d_idx,0:192])/np.linalg.norm(np.asarray(ot)))
except Exception as ex: print('step_full err',repr(ex)[:300])
```