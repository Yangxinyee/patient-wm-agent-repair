The graft uses two hidden units to gate a dose-innovation correction, and we have factual caches from 200 training stays. The synthetic contrasts are edit-design probes, not evidence of true effects. I’ll measure whether the current decoder responds coherently to those contrasts and inspect the protected feature space before fitting a replacement.

```python
print("=== SYNTHETIC CONTRAST TARGETS ===")
for label, a in (("target", sc_y), ("multiplier", sc_mult)):
    a = np.asarray(a)
    print(label, "shape", a.shape, "quantiles", np.quantile(a, [0, .1, .5, .9, 1]))
    print("first 8 / midpoint 8", a[:8], a[len(a)//2:len(a)//2+8])

old_proj = sc_Q.T @ sc_old_w
pred_old = sc_D @ old_proj
for sl in (slice(0, 78), slice(78, 156)):
    print("contrast group", sl.start,
          "target mean", float(np.mean(sc_y[sl])),
          "decoder mean", float(np.mean(pred_old[sl])),
          "target/decoder SD", float(np.std(sc_y[sl])), float(np.std(pred_old[sl])))
print("projection residual norm", float(np.linalg.norm(sc_old_w - sc_Q @ old_proj)))
print("covariance eigenvalue quantiles", np.quantile(np.linalg.eigvalsh(sc_cov), [0, .1, .5, .9, 1]))

print("\n=== CURRENT DECODER CPU CHECKS ===")
for key, cache in sc_fact.items():
    r = sc_evaluate(cache, sc_old_w, sc_old_b, sc_old_scale)
    print(key, {k: v for k,v in r.items() if np.isscalar(v)},
          "original_nll", cache["original_nll"])
syn_old = {}
for mult, cache in sc_syn.items():
    r = sc_evaluate(cache, sc_old_w, sc_old_b, sc_old_scale)
    syn_old[mult] = np.asarray(r["mean"])
for mult in (.5, 2.0):
    shift = syn_old[mult] - syn_old[1.0]
    print("synthetic multiplier", mult, "mean shift", float(shift.mean()),
          "shift quantiles", np.quantile(shift, [.1, .5, .9]))

print("\n=== RECONSTRUCTION FUNCTION INDEX ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if callable(o) and hasattr(o, "__code__") and any(
        s in n.lower() for s in ("rec_", "step", "transition", "synthetic", "posterior", "memory")
    ):
        print(n, o.__code__.co_varnames[:o.__code__.co_argcount])

print("\n=== LAST GRAFT-DESIGN NOTES ===")
design_sections = []
for s in sections:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if any(t in heading.lower() for t in ("decoder", "protected", "innovation", "synthetic")):
        design_sections.append(s)
for s in design_sections[-5:]:
    print(s[:550])

print("\n=== NOREPINEPHRINE FLEXIBLE ESTIMATES ===")
print(str(ne_dml_results)[:1800])
```
