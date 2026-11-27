An important correction. Head 3's existing 48 value dimensions carry almost none of the rate: residual sd 0.69 at insulin bundles, against 0.07 from the full normalised dose token. So the "clean channel" only works if I repurpose one of head 3's value dimensions, setting its value weights to the rate decoder.

The baseline (−rbar) through head 2 is also noisy (residual sd 0.35, or 0.21 with all heads). So I will drop the separate baseline write. Instead:
- the memory holds the raw insulin-rate trace, written through the clean u-channel;
- the glucose readout subtracts a regression-based prediction of that trace from the head's state vector q.

The model's state encodes recent glucose history, so it can predict the tier-driven part. The randomised deviation stays in the readout.

For NE that trick fails: the state tracks NE levels, so the subtraction would also cancel the ×2 effect. Whether any level-responsive NE injection is affordable depends on how much the factual best predictor's NE coefficients differ from the causal ones. That is a severity-confounding question.

**Next step:** answer it on the records with an hourly regression of MAP(t+1) on MAP lags and NE-rate lags, then compare the NE coefficients with the causal direct kernel implied by my ψ. I will first look at the prepared NE frame from earlier runs.

**What would confirm or refute it:**
- Coefficients close to causal: a modest-cost NE fix is possible.
- Strongly negative coefficients at lags ≥ 2 (severity proxy): any level-based NE fix will cost factual fit, and NE must stay partial.

```python
print(ne.columns.tolist()); print(ne.head(3).to_string()[:1500])
print("psi_ne", np.round(psi_ne,3))
```