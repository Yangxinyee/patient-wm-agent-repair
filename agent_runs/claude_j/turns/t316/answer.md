The injection at that scale is too strong. The eigen-solution's direction has norm about 4,200. At the chosen scale it pushes head 2's output far outside the linear range, so the prelude norm squashes the normal slot content.

The disturbance came out far above the linear prediction: about 2.4 RMS shift in predicted glucose means and +0.0052 nats overall, which would use the whole factual budget. Per unit of injection the signal is heavily attenuated between head 2 and the value-head input, with a best single-direction SNR of only 0.017. That is consistent with the model's own drug pathway being weak.

So the amplification has to happen after the prelude norm. The candidate is a rank-1 term in `b_in`, which maps the normalised slots into the h update. It would read a direction of the normalised slots where the natural variation is small but a moderate head-2 injection shows clearly.

I'll run diagnostics on that stage:
- the natural covariance spectrum of the normalised slots;
- the best achievable whitened SNR for a moderate head-2 injection, with ‖change in normalised slots‖ around 0.5;
- the SSM gains softplus(delta_in) and a_bar;
- how close c_out is to the identity.

A whitened SNR of at least 5 at a small perturbation would make the `b_in` route viable. If even that SNR is low, I'll stop the core route and fall back to a safe head edit.

```python
Wo2=T64(W[P+'.attn.out.weight'])[:,96:144]; SO=[]; EE=[]
for s in ST:
    X=T64(X_of(s)); B=BUN[s]
    for k in range(1,len(B)):
        mn,evs=B[k]; dt=(B[k][0]-B[k-1][0])*60.0
        hp=X[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa)); e,u=bvd(evs); zt=T64(linp(X[k,960:992],'z_in'))
        tok=torch.cat([hp+SL,T64(e)[None],T64(u)[None],zt.reshape(1,192)],0); so=tlayer(tok,P)[:4]
        SO.append(so.numpy()); EE.append(lnp(so,'core.prelude_norm').numpy())
SO=np.concatenate(SO); EE=np.concatenate(EE); print('slot_out norm',round(float(np.linalg.norm(SO,axis=1).mean()),1),'perdim sd',round(float(SO.std(0).mean()),3))
CE=np.cov(EE.T); ev_=np.linalg.eigvalsh(CE); print('E cov eig q', np.round(np.quantile(ev_,[0,.05,.25,.5,1]),5))
so_t=torch.as_tensor(SO[:400]); E0=lnp(so_t,'core.prelude_norm'); A=[]
for j in range(48):
    E1=lnp(so_t+1e-3*Wo2[:,j][None,:],'core.prelude_norm'); A.append(((E1-E0)/1e-3).mean(0).numpy())
A=np.stack(A,1); CEi=np.linalg.inv(CE+1e-6*np.eye(192)); GE=A.T@CEi@A; le,Ve=np.linalg.eigh(GE)
pE=Ve[:,-1]; dE=A@pE
print('E-space SNR per unit |p|', round(float(np.sqrt(le[-1])),2),'|dE| per unit p',round(float(np.linalg.norm(dE)),4),'|Wo2 p|',round(float(np.linalg.norm(Wo2.numpy()@pE)),3))
sp=np.log1p(np.exp(T64(W['core.delta_in']).numpy())); ab=np.exp(-sp*np.exp(T64(W['core.a_in']).numpy()))
print('softplus(delta) q',np.round(np.quantile(sp,[0,.5,1]),4),'a_bar q',np.round(np.quantile(ab,[0,.5,1]),4))
Co=T64(W['core.c_out']).numpy(); print('c_out diag mean',round(float(np.diag(Co).mean()),3),'offdiag rms',round(float(np.sqrt((Co-np.diag(np.diag(Co)))**2).mean()),4), 'b_in sv', np.round(np.linalg.svd(T64(W['core.b_in']).numpy(),compute_uv=False)[[0,96,191]],3))
```