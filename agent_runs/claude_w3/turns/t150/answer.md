My old helper scores 0.10 relative error, better than my fresh 0.41-0.59 but still not exact. Before chasing details I'll check the layout of the saved state. Per the architecture, out_k should equal out_norm(mean of the 4 saved slots of h_k) exactly. I'll also check how large the z means are, since small norms would inflate relative errors. If out ≠ LN(mean h), the slot layout is wrong (for example slot-major versus dim-major), which would explain the context errors.

```python
X_ = torch.tensor(F4['x'], dtype=torch.float64)
o1 = LNd(X_[:,192:960].reshape(-1,4,192).mean(1), 'core.out_norm')
o2 = LNd(X_[:,192:960].reshape(-1,192,4).mean(2), 'core.out_norm')
rel = lambda a, b: ((a-b).norm(dim=-1)/b.norm(dim=-1)).median().item()
print('slot-major', round(rel(o1, X_[:,:192]),4), 'dim-major', round(rel(o2, X_[:,:192]),4))
print('z norm pct', np.round(np.percentile(X_[:,960:].norm(dim=1).numpy(), [10,50,90]),3), 'out norm', round(X_[:,:192].norm(dim=1).median().item(),2), 'h norm', round(X_[:,192:960].norm(dim=1).median().item(),2))
```