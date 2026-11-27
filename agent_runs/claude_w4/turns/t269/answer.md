That was another bookkeeping error: `E_ref_d` is stored as a torch tensor, so the numpy subtraction failed. To stop paying for this kind of bug I'll add one helper, `A()`, that turns any tensor or array into float64 numpy, and use it for every stored array (`E_ref_d`, `htf`, `yt`, `ot`, `d_idx`).

The validation is the same as before. R² of about 0.99 or better for y and `out` from the chain means I can design the injection on CPU. A poor fit means I fall back to empirical probes.

```python
A=lambda t: t.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(t) else np.asarray(t,dtype=np.float64)
Ep,att_=prelude_np(A(TOK))
print('E vs E_ref_d rel err %.4f'%(np.linalg.norm(Ep-A(E_ref_d))/np.linalg.norm(A(E_ref_d))), ' head1 slot->u att mean %.4f'%att_[:,1,:4,5].mean())
dix=A(d_idx).astype(int); yt_=A(yt).reshape(2319,4,192); ot_=A(ot)
print('ot vs states out rel %.4f'%(np.linalg.norm(ot_-xo[dix,0:192])/np.linalg.norm(ot_)), ' yt vs states h rel %.4f'%(np.linalg.norm(yt_.reshape(2319,-1)-xo[dix,192:960])/np.linalg.norm(yt_)))
try:
    r=step_full(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),torch.tensor(A(htf),dtype=torch.float32))
    rr=r if isinstance(r,(tuple,list)) else [r]; print(type(r),[tuple(getattr(t,'shape',())) for t in rr])
    yv=A(rr[0]).reshape(2319,4,192); ov=A(rr[1]).reshape(2319,192)
    print('R2 y %.4f out %.4f'%(r2(yv,yt_),r2(ov,ot_)))
except Exception as ex: print('step_full err',repr(ex)[:400])
```