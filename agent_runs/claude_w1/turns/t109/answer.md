Two findings shape the design.

1. **Fast state turnover.** Each core step keeps only about 36% of the decayed state (`abar` ≈ 0.29–0.44). Persistence over hours must therefore come from the model's own write-back loop, not from slow channels.
2. **Low-rank state.** h is effectively low-rank: 133 of its 192 principal directions have sd below 0.5, while the top ones reach 155.

Building a self-sustaining dose memory inside h would need a precise feedback loop, which is fragile. Before choosing, I want the best achievable noise floor for a head-level readout. That is a linear decoder with unit response to a dose change but minimum variance on factual states (a "minimum-variance distortionless" decoder). For that I need factual head-input samples to compare with the perturbed ones. I'll first save the new findings to the notebook, then locate the training DataFrame and time `gen_samples(pert=False)` on a few stays.

```python
import pandas as pd
with open('/work/notes.md','a') as f:
    f.write("""
## Session 3 (runs 94-99)
- samp_pert.pkl = list of tuples (code, stay, hour, prev_val, cur_val, dose_lag1, lag2, lag3, with_code[192]) from gen_samples(df,pert=True), 1500 train stays, 21456 finger + 17722 art rows.
- Perturbed-dose decodability from head input (ridge, CV): ins lag1 R2 .70 (resid sd .78 u/h), lag2 .61, lag3 .57; NE lag1 .73 (resid .21). prev value R2 1.0 (register).
- Core dynamics: abar=exp(-softplus(delta_in)exp(a_in)) ~0.36 per bundle (0.29-0.44) -> state refreshed each step; persistence via loops.
  flow.a exp(a) 1.6e-8..1.8e-3 (per s). h is low-rank: slot0 PCA sd top 155,95,58; 133/192 PCs sd<0.5. No dead channels (min ch sd 2.7). 4 slots ~identical stats.
- Key theory: correct edit = add sum_k c_k * nu_{t-k}, nu = dose - E[dose|readings] (policy), c_k = missing direct coef from residual regression of
  (reading - model mean) on lagged nu. Raw-dose amplification would bias factual preds (doses track G) -> must centre on policy.
""")
dfs = sorted([(k, v.shape) for k, v in globals().items() if isinstance(v, pd.DataFrame)], key=lambda x: -x[1][0])
print(dfs[:12])
```