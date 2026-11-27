The random-direction channel is harmless but nearly invisible. The factual value NLL changes by only -0.00007 and predicted means move by under 0.05 mg/dL. At the head, though, the transmitted change is about 0.006 per U/h in with_code, and lags 1–3 explain only 18% of it. A random direction in the pre-norm slot residual (norm about 442) is diluted after the prelude norm and the state update, before it ever reaches `out`.

So the direction must be chosen for transmission. The linear part of the route from the prelude to the next state is:
- prelude norm gain divided by the residual spread (σ ≈ 32);
- b_in, scaled per dimension by softplus(delta_in);
- c_out, then along the coda's residual stream into y and `out`.

Taking the top right-singular vector of that product as the write direction should multiply the transmitted signal by the top singular value, compared with the about 1 of a random direction.

This run prints the shapes of the relevant tensors, computes the singular values of the transfer matrix, and saves the top directions. If the top singular values are 20 or more, a channel with c about 7–20 should reach the head with useful strength, which I'll test next. If they are near 1, this linear route cannot carry the signal and I'll need a different entry point.

```python
for n_ in ['core.delta_in','core.a_in','core.b_in.weight','core.c_out.weight','core.prelude_norm.weight','core.prelude_norm.bias','core.out_norm.weight','query_norm.weight','core.slots','core.flow.a']:
    print(n_, SHAPES.get(n_))
sp_=F.softplus(W['core.delta_in']).numpy(); gpre=W['core.prelude_norm.weight'].numpy(); bin_=W['core.b_in.weight'].numpy(); cout=W['core.c_out.weight'].numpy()
print('sp stats',np.round([sp_.min(),np.median(sp_),sp_.max()],3),'gpre stats',np.round([gpre.min(),np.median(gpre),gpre.max()],3))
spv=sp_.reshape(-1) if sp_.size==192 else sp_.mean(0)
T=cout@np.diag(spv)@bin_@np.diag(gpre)/32.0
Pm=np.eye(192)-np.ones((192,192))/192
U_,S_,Vt_=np.linalg.svd(T@Pm)
print('top singular values of transfer', np.round(S_[:10],2), 'median', np.round(np.median(S_),3))
# random-direction gain for comparison
print('random dir gain', np.round(np.linalg.norm(T@Pm@vs),3))
np.save('/work/transfer_T.npy',T)
```