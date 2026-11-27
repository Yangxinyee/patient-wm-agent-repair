The alignment check passed (100%), but `ctx_of` takes a row index, not a state vector. Rather than guess which array it indexes, I'll compute `ctx` directly from the documented formula, `out + out_norm(mean_slots(h·decay)) + gap(log dt/10)`.

To check it, I'll rebuild q as `query_norm(ctx + z_head(prior mean) + type_emb[1])` and compare it with the stored exact q (`Q99`). After that check, the same run does the z-surprise regression and the propagation gains from the previous plan.

The test is unchanged. The plan holds if z surprise is substantially explained by glucose and MAP surprises and those surprises propagate to the next prediction with a sizeable gain. Otherwise I switch to the prelude attention route.

```python
def LNnp(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*w+b
onw=load('core.out_norm.weight').numpy();onb=load('core.out_norm.bias').numpy()
g0w=load('gap.0.weight').numpy();g0b=load('gap.0.bias').numpy();g2w=load('gap.2.weight').numpy();g2b=load('gap.2.bias').numpy()
fl=np.exp(load('core.flow.a').numpy())
def ctx_vec(rows,dts):
    xr=X2[rows]; o=xr[:,:192]; h=xr[:,192:960].reshape(-1,4,192)*np.exp(-dts[:,None,None]*fl[None,None,:])
    gg=gelu_np(np.log(dts)[:,None]/10*g0w[:,0][None,:]+g0b)@g2w.T+g2b
    return o+LNnp(h.mean(1),onw,onb)+gg
CTr=ctx_vec(Rk,dtr)
zh_w=load('z_head.weight').numpy();zh_b=load('z_head.bias').numpy();te=load('type_emb.weight').numpy()
qnw=load('query_norm.weight').numpy();qnb=load('query_norm.bias').numpy()
W0=load('z_prior.0.weight').numpy();b0=load('z_prior.0.bias').numpy();Wp2=load('z_prior.2.weight').numpy();bp2=load('z_prior.2.bias').numpy()
zpr=gelu_np(CTr@W0.T+b0)@Wp2.T+bp2
qa=LNnp(CTr+zpr[:,:32]@zh_w.T+zh_b+te[1],qnw,qnb); print('q err half0',np.abs(qa-Q99).max().round(4))
qb=LNnp(CTr+zpr[:,32:]@zh_w.T+zh_b+te[1],qnw,qnb); print('q err half1',np.abs(qb-Q99).max().round(4))
```