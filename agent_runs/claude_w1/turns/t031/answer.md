The tool's mean_shift is the pooled mean over reached stay-hours: −1.448 matches the pooled figure, not the per-stay average. On the same rows, my estimated truth is far larger than what the model predicts:

| Edit | Model | Estimated truth |
|---|---|---|
| Insulin ×2 | −1.45 | −14.2 |
| Insulin ×1/2 | +0.82 | +7.1 |
| Norepinephrine ×2 | +0.15 | +4.8 |
| Norepinephrine ×1/2 | −0.41 | −2.4 |

So the model's insulin effect is about 10× too weak, and its norepinephrine effect about 15–30× too weak.

The dose noise is random, so the true effect is also part of the best factual predictor. A model that under-uses doses should leave residuals correlated with the dose shocks. Fixing the effect should then keep the factual NLL the same or improve it slightly: I estimate about 0.008 nats per glucose reading at stake.

To plan the edit I need to know where dose information flows. Next I'll list the tensor names and shapes: action encoder, embedding tables, core, heads. Then I'll record the forward-pass findings in the notebook.

```python
for n in NAMES:
    print(n, SHAPES[n])
with open('/work/notes.md','a') as f:
    f.write("""
## Run 24-29 + fwd 1 (dose_edit original, tuning setA = /work/setA.json, 100 stays)
- RD at insulin thresholds 180/220 agrees with noise-based estimates (-1.9..-2.6 per u/h at k=1) -> no hidden confounding.
- dose_edit rows = stay-hours 13-24 where the record HAS the outcome reading (finger glucose / map_arterial); reached = hour > first dose hour>=12.
  mean_shift = POOLED mean over reached stay-hours.
- ORIGINAL model on setA: ins x2 -1.448, ins x1/2 +0.824, ne x2 +0.152, ne x1/2 -0.411.
- My estimated TRUTH on same rows (deconvolved h): ins x2 -14.17, ins x1/2 +7.09, ne x2 +4.82, ne x1/2 -2.41.
  => model's effects ~10x (insulin) and ~30x (NE) too small.
""")
```