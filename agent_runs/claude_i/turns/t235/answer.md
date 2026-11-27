The z check never ran properly: `zpost2` returns a tuple (probably mean and log-sd), and my error print fired for every bundle. That cost no information, but I will avoid printing inside loops from now on. I will take the first element of the tuple and repeat the check for atom offsets 0, +1 and −1. If offset 0 gives errors near 0, the pipeline is confirmed.

```python
res={}
for off in (0,1,-1):
    er=[]
    for k in ksv[:30]:
        k=int(k); o_p,h_p=prev64(k)
        ctx=ctx_of_state(o_p[None],h_p[None],np.array([dtl[k]])).reshape(-1)
        e,u=eu2(atoms_by_bundle[k+off]); e=torch.as_tensor(e,dtype=T); u=torch.as_tensor(u,dtype=T)
        r=zpost2(ctx[None],e[None],u[None]); z=(r[0] if isinstance(r,tuple) else r).reshape(-1)[:32]
        zt=X64[k,960:]; er.append(float((z-zt).norm()/zt.norm()))
    res[off]=np.round(np.percentile(er,[50,90]),5)
print(res, 'tuple len', len(r) if isinstance(r,tuple) else 1, [tuple(x.shape) for x in r] if isinstance(r,tuple) else '')
```