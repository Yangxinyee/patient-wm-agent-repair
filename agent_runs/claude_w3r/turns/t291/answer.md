The pair-based gated linear decoder works mechanically:
- The FFN output matches the ideal decode with no bias (noise 0.15 insulin, 0.06 norepinephrine).
- It is exactly 0 at every u-absent bundle.
- The E-channel reads it with noise ≤0.03.

The weak point is the linear decode itself. Insulin saturates near 4.9, so doubled high rates are underestimated by about 1.8. At norepinephrine-only bundles the insulin read is noisy (sd 0.8). Since calibration is end-to-end, I accept this for a first full build and refine later if runs allow.

Stage 3 next: a rank-1 `core.b_in` edit per drug that writes the E-channel into h_mid along y-directions the heads can read. For that I need the name of the input gate `sp`, the slot and flow parameters, and the code-embedding tensor (for code-gating in the value head).

```python
skip=('prelude','coda','blocks','attr_heads.rate','attr_heads.status')
for n in NAMES:
    if (n.startswith('core.') and not any(s in n for s in skip)) or 'code' in n or ('emb' in n and 'core.' not in n): print(n,SHAPES[n])
print([k for k in W.keys()][:40] if 'W' in globals() and isinstance(W,dict) else 'no W dict')
```