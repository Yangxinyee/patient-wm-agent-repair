The model ignores dose magnitude almost entirely when readings are held at their recorded values. Per unit/h of insulin, the glucose sensitivity is −0.36 at lag 1, −0.04 at lag 2 and ≈ 0 after that. For norepinephrine, the MAP sensitivity is about −0.05 to −0.09 per mcg/kg/min at every lag. The records imply a direct kernel of about −2.1, −1.5, −0.5 at lags 1–3 for insulin and about +3.4 at lag 1 for norepinephrine.

So the edit has to add dose-magnitude dependence spread over several hours. That needs a dose memory that survives several core steps (about 2 bundles per hour). Next I'll look at the core's built-in persistence structure:
- the per-dimension update factor a_bar = exp(−softplus(δ)·exp(a_in));
- the time-flow decay per hour;
- how close `c_out` is to identity;
- how much variance each state coordinate has.

Dimensions with a_bar near 1, slow decay and a near-identity `c_out` would be natural carriers. If none exist, I'll need a different route, such as carrying the effect through the model's glucose belief.

```python
sp_ = F.softplus(Po['core.delta_in']); abar_ = torch.exp(-sp_*torch.exp(Po['core.a_in']))
dec_h = torch.exp(-3600*torch.exp(Po['core.flow.a']))
print("abar quantiles", np.round(np.quantile(abar_.numpy(), [0,.1,.5,.9,1]),3))
print("time-flow decay per hour quantiles", np.round(np.quantile(dec_h.numpy(), [0,.1,.25,.5,.75,.9,1]),3))
Wc = Po['core.c_out.weight']
print("c_out diag stats", np.round(np.quantile(torch.diag(Wc).numpy(), [0,.1,.5,.9,1]),3), "offdiag abs mean", float((Wc - torch.diag(torch.diag(Wc))).abs().mean()))
sv = torch.linalg.svdvals(Wc); print("c_out singular values top/bottom", np.round(sv[:5].numpy(),3), np.round(sv[-5:].numpy(),3))
print("||c_out - I|| / ||I||", float(torch.linalg.norm(Wc-torch.eye(192))/math.sqrt(192)))
Bin = Po['core.b_in.weight']; print("b_in sv top", np.round(torch.linalg.svdvals(Bin)[:5].numpy(),2))
hs_std = HS.std(0); hs_mean = HS.mean(0)
print("HS per-coord std quantiles", np.round(np.quantile(hs_std.flatten().numpy(), [0,.05,.25,.5,.75,.95,1]),3))
print("HS per-coord |mean| quantiles", np.round(np.quantile(hs_mean.abs().flatten().numpy(), [0,.05,.25,.5,.75,.95,1]),2))
# combined per-hour persistence for 2 steps if c_out ~ I: (abar*dec)^...
pers = (abar_**2 * dec_h).numpy(); print("abar^2*dec_h quantiles", np.round(np.quantile(pers,[0,.5,.9,.99,1]),3), "argmax dims", np.argsort(-pers)[:8])
```