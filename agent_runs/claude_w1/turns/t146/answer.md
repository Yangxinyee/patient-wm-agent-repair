Run 136 has two findings.
- **Insulin innovations:** `NUA` is not aligned with the setA insulin doses (correlation 0.01), so the glucose regressions in that run are invalid.
- **MAP regressions:** these are valid (`nuN` checks out).
  - The real model's MAP residuals load positively on lagged norepinephrine innovations: [3.7, 1.6, 1.1]. This matches the training-set estimate.
  - The decoded dose minus the expected rate loads negatively at lag 1: −1.5 ± 0.75.
  - So the wrong-sign problem is not a replica artefact. The decoder's factual error is correlated with the model's residual. A naive decoded-dose correction with the right causal gain would cost factual fit. My estimate is on the order of the whole 0.005-nat budget once insulin and extra lags are added.

The fix needs a decoder with unit dose response that is also chosen to minimise factual harm. That is a constrained regression, and it needs many exact factual residuals. An exact local head replica would give me those for unlimited training stays. It would also let me simulate rollouts and check factual NLL locally.

My head replica is off by about 2 mg/dL, so I'll re-read the head section of ARCHITECTURE.md to find what I have wrong. What I need: the exact definitions of the query, `last_numeric`, `last_status`, the type embedding and the transition table.

```python
import re
T_ = txt
for kw in ['attr_heads', 'value_bin', 'last_numeric', 'last_status', 'query', 'type_emb', 'embed_transition', 'register', 'readout', 'head']:
    idxs = [m.start() for m in re.finditer(kw, T_)]
    print(kw, len(idxs), idxs[:8])
st_ = T_.find('last_numeric')
a_ = max(0, T_.rfind('\n#', 0, st_)); print(T_[a_:a_+3800])
```