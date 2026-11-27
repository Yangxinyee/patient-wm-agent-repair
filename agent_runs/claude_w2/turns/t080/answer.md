The core is a pre-norm transformer stack, which rules out a cheap dedicated memory channel in h:
- a_bar keeps only about 0.31–0.46 of h per bundle.
- Memory is carried by the prelude, blocks and coda, which re-write h at every step.
- `c_out` mixes all dimensions.
- No dimension of h is dead: the lowest per-dimension std is about 1.8, against a median of about 5.5.

So a hand-built persistent dose memory would mean surgery on dozens of matrices, at serious risk to the 0.005-nat factual budget.

**What this implies for the plan.** The workable route is a correction in the value head, driven by dose information the state already carries. The current dose decodes from `out` with R² ≈ 0.8. The longer-lag accumulation should come from the model's own propagation of its drawn glucose values. Below, φ is how much of a glucose deviation the patient's own dynamics keep from one hour to the next.
- If glucose itself mean-reverts with φ ≈ 0.8 per hour, my structural impulse response implies one-step dose coefficients (given the previous reading) of about [−2.0, −0.9, −0.2, 0, …].
- For norepinephrine with φ ≈ 0.75 they would be about [+3.6, 0, …].
- A lag-1 or lag-2 head correction would then be enough, provided the model's glucose persistence matches the records.

I will test this with regressions: the records' one-step conditional mean (local, large n) against the model's predicted means.

**Next step: build a clean head implementation.** For that I need the exact head tensors and the vocabulary layout (bin edges and centres per code, normalisation). This run lists all non-core tensors and summarises the vocab. I will then rebuild the head with 4 prior-z draws averaged, as the architecture specifies. That should close the 0.02–0.07 NLL gap from run 71. If it does not, the remaining mismatch is in the view or ctx construction.

```python
others = [n_ for n_ in NAMES if not n_.startswith('core.')]
print(len(others))
print('; '.join(f"{n_} {tuple(SHAPES[n_])}" for n_ in others))
print(list(VOCAB.keys()))
print({k_: (type(v_).__name__, (list(v_.keys())[:12] if isinstance(v_, dict) else (len(v_) if hasattr(v_,'__len__') else v_))) for k_, v_ in VOCAB.items()})
print('config:', VOCAB.get('config'))
```