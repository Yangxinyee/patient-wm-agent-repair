The u-channel now works. u[4]−u[2] tracks 0.2×rate (mean error 0.0002, max 0.16) and stays ≤0.08 at norepinephrine-only bundles. The memory increment is linear in the start rate (slope 0.49 per unit/h, intercept 0.02, R² 0.963), with essentially zero writes at non-drug, norepinephrine-only and stop-only bundles. The R² below the decoder's 0.998 is consistent with multiplicative noise from the two norm divisions. The write path costs +0.0016 nats on the 300-stay set, which leaves room for the readout and possibly norepinephrine.

Next is the readout, which needs the value head's `with_code` vectors at glucose_finger predictions. lib3 has `capture_wc` and `mean_ablate` helpers for exactly this. I'll print their source (lib3 lines 4–49) to reuse them correctly.

```python
L=open('/work/lib3.py').read().splitlines()
for i,l in enumerate(L[3:49]): print(i+4, l[:150])
```