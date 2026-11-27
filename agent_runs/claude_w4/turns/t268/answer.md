My replica assumed an FFN pre‑norm called `norm2`, but the prelude only has `norm1` and `norm3`. The `norm2` slot is used for cross‑attention in the recurrent blocks, and the prelude has none, so its FFN uses `norm3`.

I'll fix that and rerun the same validation:
- prelude E against my earlier reference;
- the chain through `step_full` against the exact GPU states (`yt`, `ot`).

The test is unchanged. R² of about 0.99 or better for y and `out` makes the CPU chain usable for designing the injection. A low R², or an error from `step_full`, sends me to empirical calibration.

```python
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
    h2=LNn(x,W['core.prelude.0.norm3.weight'],W['core.prelude.0.norm3.bias'])
    x=x+G(h2@W['core.prelude.0.ffn.0.weight'].T+W['core.prelude.0.ffn.0.bias'])@W['core.prelude.0.ffn.2.weight'].T+W['core.prelude.0.ffn.2.bias']
    return LNn(x[:,:4],W['core.prelude_norm.weight'],W['core.prelude_norm.bias']), s
Ep,att_=prelude_np(TOK)
print('E vs E_ref_d rel err %.4f'%(np.linalg.norm(Ep-E_ref_d)/np.linalg.norm(E_ref_d)), ' head1 slot->u att mean %.4f'%att_[:,1,:4,5].mean())
def r2(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()
try:
    r=step_full(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),torch.tensor(np.asarray(htf),dtype=torch.float32))
    print(type(r), [getattr(t,'shape',None) for t in (r if isinstance(r,(tuple,list)) else [r])])
    yv=np.asarray(r[0]).reshape(2319,4,192); ov=np.asarray(r[1]).reshape(2319,192)
    print('R2 y %.4f out %.4f'%(r2(yv,np.asarray(yt)),r2(ov,np.asarray(ot))), ' ot vs states out rel %.4f'%(np.linalg.norm(np.asarray(ot)-xo[d_idx,0:192])/np.linalg.norm(np.asarray(ot))))
except Exception as ex: print('step_full err',repr(ex)[:300])
```