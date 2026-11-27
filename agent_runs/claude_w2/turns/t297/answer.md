The build failed on two counts:
1. The factual shift per row has sd 24 against an intended 8.5.
2. The factual glucose bin NLL gets worse by +0.33.

The cause is the shape of the edit: a linear logit tilt along fixed bin centres. The predicted distributions are sharp with heavy tails and range over 72–271 mg/dL, so a fixed tilt pumps up far bins and the special bin 0 instead of shifting the distribution. The gate itself worked (glucose ≥ 16, MAP ≤ −16, residual < 0.007).

A better shift uses the model's own machinery. Make the value MLP see the "last value of this code" as if it were lower by s. That is a rank-1 change to its first layer: W1 ← W1 − (W1·u)·wᵀ/c, where u is the with_code direction of the z_last input. The model already knows how to move its whole predicted distribution with the last value, so its shape is preserved.

To build it I need the exact head code: which tensor maps the 8 last-value features into with_code, and the column order. I'll print `head_value_logits`.

```python
src_h = open('/work/head.py').read()
i0 = src_h.find('def head_value_logits'); i1 = src_h.find('def head_probs_mc')
print(src_h[i0:i1][:1200])
```