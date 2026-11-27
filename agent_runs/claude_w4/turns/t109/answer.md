The readout biases over the real mix of drug bundles are small:
- **Insulin:** relative error about 1% for every multiplier.
- **NE:** relative error under 1% for x1/2 and x2, about 5% for x0, and −10% for x3, where mixed bundles at very high NE rates go wrong.

The per-bundle rms errors (0.18 units/h for insulin) are random and would move a single factual prediction by well under 1 mg/dL. So simple gated linear readouts of u_k are good enough to proceed.

Before writing into z_post I must be sure the hidden units I repurpose are dead at every bundle, not only at drug bundles. At measurement bundles I cannot replicate e_k exactly. But e_k is a LayerNorm output, so its contribution to a unit's pre-activation has a hard upper bound. This run computes:
- each unit's pre-activation without the e term, at drug bundles (from my exact inputs);
- the analytic worst-case e term;
- as a check, whether units already dead at drug bundles have a ctx and u part far enough below zero that even the worst-case e cannot switch them on.

```python
W0f = load('z_post.0.weight').double(); b0f = load('z_post.0.bias').double()
W0c, W0e, W0u = W0f[:, :192], W0f[:, 192:384], W0f[:, 384:]
ge = load('encode.norm.weight').double(); be = load('encode.norm.bias').double()
wg = W0e * ge; wgc = wg - wg.mean(1, keepdim=True)
ebound = np.sqrt(192)*wgc.norm(dim=1) + W0e @ be
XIN3d = torch.as_tensor(np.asarray(XIN3), dtype=torch.float64)
pre_noe = XIN3d[:, :192] @ W0c.T + XIN3d[:, 384:] @ W0u.T + b0f
pre_full = XIN3d @ W0f.T + b0f
cu = torch.as_tensor(np.asarray(u_empty), dtype=torch.float64).ravel()
pre_meas_ctxu = XIN3d[:, :192] @ W0c.T + (W0u @ cu) + b0f
UB = pre_meas_ctxu.max(0).values + ebound
cand = torch.argsort(UB)[:16]
print('units with UB<-3:', int((UB < -3).sum()), ' UB<0:', int((UB<0).sum()))
print('best UB', np.round(UB[cand].numpy(),1)); print('their max pre at drug bundles', np.round(pre_full.max(0).values[cand].numpy(),1)); print('ids', cand.numpy())
```