Calling the old helper on the full table with z set to zero ran. For one reading, its bin probabilities and mine differ smoothly: mine are wider and shifted up. Its mean (172.5) and the tool's (175.7) also differ by 3 mg/dL, so it probably used z = 0 rather than the prior mean, and it is not an exact reference either. I will stop trying to match its internals.

For designing the edit, what matters is how the model's state responds to dose changes and how that response reaches the glucose predictions. The exact head details matter less, since I will validate any edit with the tool.

The edit I am aiming for adds the missing dose effect, Σ_k D_k·(dose at lag k − the dose the policy gives at that glucose). That form improves the factual fit and gives correct answers to the dose questions. It needs a state component that remembers recent doses and stays linear in the dose.

So the key structural question is whether h carries information forward almost linearly from step to step: is c_out near identity, and how fast does each dimension decay? Next I will look at c_out's closeness to identity, the per-dimension decay timescales, the per-step retention a_bar, and the size of the coda's residual update relative to h.

How I will read it:
- **c_out ≈ identity, small residual updates, some slow dimensions:** a dose memory injected into h would persist almost linearly, and I can build the channel there.
- **Otherwise:** I need a different carrier for the dose memory.

```python
Co = WT['core.c_out.weight'].numpy()
print('c_out: ||C-I||/||I||', round(np.linalg.norm(Co-np.eye(192))/np.sqrt(192),3), 'diag mean', round(np.diag(Co).mean(),3), 'offdiag rms', round(np.sqrt((Co-np.diag(np.diag(Co)))**2).mean()*192/191,4))
sv_ = np.linalg.svd(Co, compute_uv=False); print('c_out sv range', np.round(sv_[[0,10,50,100,191]],3))
fa_t = WT['core.flow.a'].numpy(); tau_h = 1/np.exp(fa_t)/3600
print('flow timescales (h): quantiles', np.round(np.quantile(tau_h,[0,.1,.25,.5,.75,.9,1]),3))
sp_ = np.log1p(np.exp(WT['core.delta_in'].numpy())); ab_ = np.exp(-sp_*np.exp(WT['core.a_in'].numpy()))
print('a_bar quantiles', np.round(np.quantile(ab_,[0,.1,.25,.5,.75,.9,1]),4), 'softplus(delta) q', np.round(np.quantile(sp_,[0,.5,1]),3))
# residual magnitudes on a typical step
k_ = 40; hf = HHt[k_-1] * torch.exp(-dts[k_]*torch.exp(WT['core.flow.a']))
print('typical |h| per slot', np.round(HHt[k_].norm(dim=1).numpy(),1), '|hf|', np.round(hf.norm(dim=1).numpy(),1))
```