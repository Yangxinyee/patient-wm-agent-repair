Two findings from the decoder fits.

- **Rate from u is easy.** A linear readout from u recovers the insulin rate with R² 0.998 on recorded bundles, and NE with R² 0.995. On synthetic edited atoms (x0 to x3) it still reaches R² 0.99. Errors grow only at x3: insulin RMSE 0.41 U/h, NE 0.11.
- **Policy from ctx is not linear.** A linear readout from ctx predicts the recorded rate with R² only 0.55 for both drugs, against 0.82–0.94 for my policy GBM. A linear π̂ would leave a lot of confounded policy signal in the "residual". So π̂ has to be nonlinear: many gelu units inside z_post, fitted by ridge regression on fixed random features.

It also has to be gated. The subtraction should happen only when the drug is present, which I can do with a large negative bias driven by a "drug present" readout from u.

One risk must be settled before I build this into weights: does the core use the z posterior mean or a sample? If it samples with the z_post log-variance outputs, the repurposed z token would carry noise in the evaluation's rollout, and I would need to pin those outputs very low. I'll grep ARCHITECTURE.md for how z is drawn and used.

```python
txt = open('/model/ARCHITECTURE.md').read()
keys_ = ['z_post', 'posterior', 'sample', 'z_in', 'prior', 'rollout', 'logvar', 'log_std', 'std']
lines = txt.splitlines(); shown = 0
for i, l in enumerate(lines):
    if any(k in l for k in keys_):
        print(i, l[:170]); shown += 1
        if shown >= 30: break
print('total lines', len(lines))
```