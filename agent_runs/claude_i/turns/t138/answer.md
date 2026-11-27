Most inputs agree: the previous-bundle index matches, `out` matches exactly and the normalised mean slot to 7e-5. The gap embedding does not: it differs by 0.31, and so does the z prior, which depends on it.

So the earlier head, which tracks the tool well, used a different input to the gap MLP from my log(dt)/10. ARCHITECTURE.md also writes the view's gap as `gap(log dt)`, without the /10 used for the context. The tiny 0.12 s dt difference cannot explain a 0.31 gap.

Next I will find which gap input reproduces the old head's gap vector. I will try log(dt), log(dt)/10, log1p(dt)/10, and log of dt in minutes or hours.
- **If one variant matches to ~0:** that is the view's gap input.
- **Then:** I will recheck my head's error against the tool, expecting it to fall to about the old head's level (~1.9 for glucose_finger).

```python
dt0 = float(r0['dt'])
cands = {'log/10': np.log(dt0)/10, 'log': np.log(dt0), 'log1p/10': np.log1p(dt0)/10, 'logmin/10': np.log(dt0/60)/10, 'logh/10': np.log(dt0/3600)/10,
         'logmin': np.log(dt0/60), 'logh': np.log(dt0/3600), 'log1p_min/10': np.log1p(dt0/60)/10}
for k_, x_ in cands.items():
    g_ = linw(Fnn.gelu(linw(torch.tensor([x_], dtype=torch.float32), 'gap.0')), 'gap.2').numpy()
    print(k_, round(float(np.abs(gv_[i0] - g_).max()), 5))
```