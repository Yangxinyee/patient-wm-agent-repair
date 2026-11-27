This confirms the diagnosis, and it is serious. The pre-LayerNorm encoder output x2 has a std that swings with the dose itself. For insulin-only bundles it is 34 at 0.5–1 U/h, 71 at 1.5, 19 at 2 and 13 at 3. The channel's gain is therefore ∝ 1/σ: deterministic and small-spread within a rate band (e.g. 0.197 ± 0.011) for one-drug bundles, but very noisy (± up to 100%) in two-drug bundles, where σ depends on both rates.

All dose information must cross this LayerNorm, so any dose channel must be pre-scaled by σ(bundle) in x2. For one-drug bundles that's a 1-D function of the rate, which mlp1 can produce as a piecewise-linear sum of gated GELU kinks. Two-drug bundles (28% of insulin bundles, 69% of NE bundles) would need an approximation.

Before committing to that build, I need two things:
- Whether the later LayerNorms on the path (prelude-slot norm, core `out_norm`, head `query_norm`) also have strongly varying stds. If they do, the channel would be distorted again downstream.
- How many truly unused mlp1/mlp2 units are available for the kinks.

This run measures the coefficient of variation of each LayerNorm's per-row std and counts near-dead units.

```python
def cv_(X):
    s=_lnstd(X.reshape(-1,X.shape[-1])); return '%.1f±%.1f (cv %.2f)'%(s.mean(),s.std(),s.std()/s.mean())
print('encoder x2 (drug bundles):', cv_(E0['x2']))
print('prelude slot tokens X7[:, :4] (hf+slots):', cv_(Dd['X7'][:,:4]))
print('core y (h after update, insulin bundles):', cv_(Cs['y'].reshape(-1,4,192)), ' mean-over-slots y:', cv_(Cs['y'].reshape(-1,4,192).mean(1)))
print('head base (view):', cv_(Dw['base'][::7].astype(np.float64)))
dead1=np.where(imp1<1e-3)[0]; dead2=np.where(imp2<0.02)[0]
print('mlp1 units imp<1e-3:',len(dead1),' imp<0.005:',int((imp1<0.005).sum()),'| mlp2 units imp<0.02:',len(dead2),' <0.05:',int((imp2<0.05).sum()))
```