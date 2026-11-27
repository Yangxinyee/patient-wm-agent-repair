Scaling u_k up, dropping its LayerNorm or removing its affine do not fix the 0.5 u/h case. The model still gives −1.12 where mine stays positive. So the model's representation of the insulin rate differs from mine in a way I cannot locate cheaply. I will stop trying to replicate z exactly; it has cost about 25 runs.

Plan from here:
1. Implement only the prediction heads locally. They read the stored out and h, so they can be checked against the model's factual predicted means.
2. Design the edit at the value head, with any extra dose channel checked empirically through states and factual requests on the edited model.

First I record the findings and the pivot in the notebook, so they survive the next restart.

```python
notes_add = """
## Session 2 (restart 2) - local re-implementation attempts (runs 39-64)
- ARCH details confirmed: GELU (exact) everywhere; gap input = log(dt_seconds)/10; states out == core.out_norm(mean_slots(h)) exactly (h = post-coda y).
- Bundling (build_bundles) matches model exactly (times, n_atoms incl. END atoms).
- Embedding tables: row0 = padding (zero) for ekind, text, bins, categoricals; untrained rows norm~0.28 (kq1-3, cat row1s). Used: unit rows 2-5, rate_unit rows 2-3, kq[0].
- z_post reproduction: median abs err ~0.05 (|z|~0.58) at best (END atoms via OBS encoder better: 0.015 on action bundles). Unit mapping barely matters.
- STRIKING: model posterior z dims 9 & 20 at insulin bundles are BINARY in rate: rates 1.0/1.5 -> +1.08/+0.84; rates 0.5 and >=2 -> -1.1/-1.3 (sd .01).
  My implementation gives a weaker, smooth version (0.5 -> +0.3!). Could not find cause (bins, z, END handling, pooling, LN all tested). Insulin rates in data are multiples of 0.5; rate==dose.
- Insulin rate edges have duplicates [0.5 0.5 0.5 1 1 1 1 1.5 1.5 2 2.5 3 3.5 4 5] (side=left: 0.5->1, 1->4, 1.5->8, 2->10).
- DECISION: stop exact z replication. Pivot: implement HEADS locally from states (out,h) and validate vs factual predicted means; design head-level edit + empirically verified dose channel.
- KEY DESIGN (insulin): desired added term to glucose pred at reading s+1: b1(r_s - m(g_s)) + b2(r_{s-1} - m(g_{s-1})) [+b3...], m = band-mean policy.
  Compensation -m(g_last) can live in embed_transition.value_bin (per last-bin rows). Action bundle at s:05 holds r_s (start atom) and r_{s-1} (END atom) -> lag1+lag2 available in one bundle.
  Feedback via drawn glucose carries AR(1) rho~0.75; with b1=-1.95, b2~-0.83, b3~-0.22 reproduces structural IRF.
"""
with open('/work/notes.md','a') as f_: f_.write(notes_add)
print(len(open('/work/notes.md').read()))
```