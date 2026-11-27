Excluding first bundles, the dt mismatch is negligible (0.24 s), so dt is right. The z reconstruction, though, is completely off (relative error ≈ 1), which means I used the wrong helpers. My notes say the exact versions are `zpost2` (enc2.py) and `ctx_of_state` with `gap2` (head2.py). `zpost_mean` and `gapf` in sim.py are older and may be stale. A misaligned `atoms_by_bundle` index would also produce this. I will test the float64 helpers, also with atom index offsets of ±1. If zpost2 with offset 0 gives an error near 0, the pipeline is fixed and the problem was the old z/gap code. If only an offset matches, the bundle indexing is shifted.

```python
X64=torch.tensor(np.asarray(X3),dtype=torch.float64)
def prev64(k):
    if first[k]: return torch.zeros(192,dtype=T), torch.zeros(4,192,dtype=T)
    return X64[k-1,:192], X64[k-1,192:960].reshape(4,192)
res={}
for off in (0,1,-1):
    er=[]
    for k in ksv[:30]:
        k=int(k); o_p,h_p=prev64(k)
        ctx=ctx_of_state(o_p[None],h_p[None],np.array([dtl[k]])).reshape(-1)
        e,u=eu2(atoms_by_bundle[k+off]); e=torch.as_tensor(e,dtype=T); u=torch.as_tensor(u,dtype=T)
        try: z=zpost2(ctx[None],e[None],u[None]).reshape(-1)[:32]
        except Exception as ex: print('zpost2 err',repr(ex)[:200]); z=torch.zeros(32,dtype=T)
        zt=X64[k,960:]; er.append(float((z-zt).norm()/zt.norm()))
    res[off]=np.round(np.percentile(er,[50,90]),5)
print(res)
```