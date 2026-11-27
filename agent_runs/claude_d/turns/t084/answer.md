The norepinephrine check came out the same way as insulin. One hour after a dose change the map_arterial residuals carry +3.3 ± 1.3 mmHg per unit rate of unexplained effect, about the full estimated effect, so the model barely responds there. At lag 2 the figure is also +3.3 ± 1.3, with lags 3–4 near zero within noise. The residual sd is 5.2 mmHg.

So both actions are under-used in factual prediction too, and a centred correction should improve factual fit. A rough budget estimate rules out an uncentred correction: added prediction bias of about (correction gain) × (expected rate) after every recorded action would cost about 0.008 nats per reading, over the 0.005 limit.

A centred feature needs a place where the pre-bundle context and the action meet. That place is the hidden layer of `z_post` (input [ctx, e_k, u_k]). Hidden units that feed only the log-sd outputs are free to reuse, because inference uses only the posterior mean. Next I'll run three checks: whether `out` equals LayerNorm(mean over slots of h), which tests my norm conventions; the decay rates; and how each `z_post` hidden unit's output weights split between the mean and log-sd halves.

```python
Wn=load('core.out_norm.weight').numpy() if hasattr(load('core.out_norm.weight'),'numpy') else np.asarray(load('core.out_norm.weight'))
T=lambda n: np.asarray(load(n).numpy() if hasattr(load(n),'numpy') else load(n),dtype=np.float64)
def LN(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*w+b
H4=X3[:,192:960].reshape(-1,4,192).astype(np.float64); OUT=X3[:,:192].astype(np.float64)
o_hat=LN(H4.mean(1),T('core.out_norm.weight'),T('core.out_norm.bias'))
print('out check max abs err',np.abs(o_hat-OUT).max().round(5),' out scale',np.abs(OUT).mean().round(3))
fa_=T('core.flow.a'); print('exp(flow.a) range',np.exp(fa_).min(),np.exp(fa_).max(),' tau(s) min/max',(1/np.exp(fa_)).min().round(1),(1/np.exp(fa_)).max().round(1))
W2=T('z_post.2.weight'); print('z_post.2 row norms mean-half',np.linalg.norm(W2[:32],axis=1).mean().round(3),'logsd-half',np.linalg.norm(W2[32:],axis=1).mean().round(3))
cn=np.linalg.norm(W2[:32],axis=0); cl=np.linalg.norm(W2[32:],axis=0); print('hidden units: outgoing-to-mean norm quantiles',np.quantile(cn,[0,.05,.1,.25,.5]).round(4),' n<0.01:',(cn<0.01).sum(),' n<0.05:',(cn<0.05).sum())
```