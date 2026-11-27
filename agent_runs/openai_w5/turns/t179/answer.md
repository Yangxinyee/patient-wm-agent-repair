The intended sensor gates are near 8, firmly in GELU’s linear range, so gate clipping is unlikely to explain the asymmetry. Dose recovery is imperfect, especially for norepinephrine. I’ll now hold history fixed and test the softmax decoder: asymmetric shifts there would implicate the logit tilt rather than feedback alone.

```python
from scipy.special import softmax

logits155 = array154(_r149_logits32).astype(np.float64)
P155 = softmax(logits155, axis=-1)
print("=== FIXED-HISTORY, EXACT-DOSE DECODER RESPONSE ===")
for j155, sp155 in enumerate(_r148_specs):
    own155 = _r148_event_array == sp155["event"]
    d155 = _r147_dose[:, sp155["drug"]]
    active155 = own155 & (d155 > 0) & _r147_test.astype(bool)
    c155 = np.r_[0.0, VOCAB["value_bin_centres"][sp155["event"]]]
    tilt155 = sp155["tilt"] * _r149_strengths[j155]
    mu155 = (P155 * c155).sum(axis=-1)
    deriv155 = (
        P155 * (c155 - mu155[..., None]) * tilt155
    ).sum(axis=-1).mean(axis=0)
    print(sp155["event"], "active holdout readings", int(active155.sum()))
    print("unit-dose derivative q10/q50/q90:",
          np.round(np.quantile(deriv155[active155], [.1, .5, .9]), 4))
    print("dose-weighted derivative:",
          round(float(np.sum(deriv155[active155]*d155[active155]) /
                      np.sum(d155[active155])), 4))
    print("calibration coefficient times strength:",
          sp155["beta"] * _r149_strengths[j155])
    for mult155 in [0.0, 0.5, 1.5, 2.0, 3.0]:
        dl155 = (mult155 - 1.0) * d155
        counter155 = softmax(
            logits155 + dl155[None, :, None]*tilt155[None, None, :], axis=-1
        )
        shift155 = (
            (counter155*c155).sum(axis=-1)-mu155
        ).mean(axis=0)
        linear155 = dl155*deriv155
        print(" multiplier", mult155,
              "finite mean", round(float(shift155[active155].mean()), 4),
              "linear mean", round(float(linear155[active155].mean()), 4))

print("=== RECONSTRUCTION / TRANSPORT CACHE NAMES ===")
print([k for k, v in snapshot150
       if re.search(r"^(_r13[89]_|_r14[01]_|cache|states|model)", k)][:120])

with Path("/work/notes.md").open("a") as f:
    f.write("""
## Runs 153–155: asymmetry diagnosis
- Intended sensor control gates are approximately 8: GELU gate clipping is ruled out on saved factual features.
- Fixed-dose perturbations affect only the intended reserved neuron; ordinary hidden neurons are unchanged.
- Held-out dose recovery RMSE: finger .144, lab .135, arterial .116, cuff .072.
- Testing finite softmax response separately from transport and recurrent feedback; no new weights edited.
""")
```
