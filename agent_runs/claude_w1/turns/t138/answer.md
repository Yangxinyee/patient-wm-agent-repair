The previous reply was interrupted. I'll carry on from the run 128 output and keep these replies focused on the technical work.

**What run 128 showed.**
- My causal estimates are in the kernel:
  - fixed-dose impulse responses `hI_full` (insulin) and `hN_full` (norepinephrine);
  - local-projection totals `TGi`/`TGn`;
  - dose persistence `TIi`/`TIn`.
- `keep`/`mu_all`/`res_all` hold replica-model means and residuals for 153,015 factual readings from 6,000 training stays.
  - glucose_finger residual mean is −3.3 mg/dL, which suggests my approximate head replica is biased;
  - MAP residual mean is −0.4.
- There is also a 35,007-row block whose names suggest a factual-versus-perturbed-dose comparison of head inputs: `xF`, `xP`, `dX`, `LdD`, `Lnu`, `res`.
- `bet = [-1.77, 0.66, 0.08, 0.03]` looks like a residual-on-innovation regression, but I can't confirm that from the notebook.

**What I conclude.** The theory in the notes is right. The edit should add c_k·(dose − policy-expected dose) for lagged doses, not c_k·dose. A model that under-uses the dose compensates through the reading itself. Only the innovation form fixes both the dose coefficient and that compensation, and so keeps the factual fit.

**Next step.** I'll identify the 35,007-row block and check it against my estimates:
- the policy notes;
- glucose and MAP bin edges, to see whether the policy thresholds line up with bins;
- the value-head tensor shapes;
- summaries of `LD`/`LdD`/`Lnu`/`res`;
- a residual-on-lagged-innovation regression.

If that regression shows large negative insulin coefficients at lags 1–3, the missing-effect story is confirmed. If they are near zero, the problem sits elsewhere, perhaps in the rollout dynamics.

```python
i0_ = s.find('INSULIN POLICY'); print(s[i0_:i0_+700])
for c_ in ['glucose_finger','map_arterial']:
    vb_ = VOCAB['bins'][c_] if 'bins' in VOCAB else None
    print(c_, {k_: (np.round(np.asarray(v_),1).tolist() if hasattr(v_,'__len__') else v_) for k_,v_ in (vb_.items() if isinstance(vb_,dict) else [])} if vb_ is not None else list(VOCAB.keys()))
print([ (n_, SHAPES[n_]) for n_ in NAMES if ('value' in n_ or 'transition' in n_ or 'last' in n_)])
for nm_ in ['LD','LdD','Lnu']:
    A__ = g[nm_]; print(nm_, 'mean', A__.mean(0).round(2).tolist()[:6], 'sd', A__.std(0).round(2).tolist()[:6])
for nm_ in ['res','yv','yhat','e','e1','curv','gl1','rs','rt','T','B','Bv']:
    A__ = np.asarray(g[nm_]); print(nm_, A__.shape, np.round([np.nanmean(A__), np.nanstd(A__), np.nanmin(A__), np.nanmax(A__)],3).tolist())
Xr_ = np.c_[np.ones(len(res)), Lnu[:, :6]]
cf_, *_ = np.linalg.lstsq(Xr_, res, rcond=None); print('res~Lnu[1..6]', cf_.round(3).tolist())
print('b', np.round(b,3).tolist())
```