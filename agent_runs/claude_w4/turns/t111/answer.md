The context predicts the recorded insulin rate with cross-validated R² of 0.56. I expect the policy ceiling (band mean of latest glucose, given its multiplicative noise) to be about 0.6–0.65, so a linear readout of ctx captures most of the predictable part.

For NE the R² is only 0.31, because the model's state retains little of the previous NE rate. For the factual-fit argument, the expectation that matters is the model's own: what it can predict from its state. So the ridge regression on ctx is the right subtraction for both drugs.

Next is the injection direction in z space. At measurement bundles the posterior mean encodes each reading's surprise, i.e. observed minus the model's predicted mean. I regress the z posterior mean on the four surprises (finger, lab, arterial, cuff) to get "z shift per mg/dL (or mmHg) of surprise". I also check whether the finger and lab directions agree, and the arterial and cuff ones, since they should if z carries the latent level. First I check the factual file's layout and how bundles align.

```python
print(F2.columns.tolist(), F2.shape); print(F2.head(2).to_string())
print('Z3', np.asarray(Z3).shape, ' hr3 meas sample', np.round(np.asarray(hr3)[:6],4))
```