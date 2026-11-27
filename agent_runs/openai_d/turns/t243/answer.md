Insulin alone adds about 0.00370 nats on this cohort, leaving little room for norepinephrine; either combined version adds about 0.0073. I’ll examine whether the corrections have systematic factual offsets and recover the existing dose-probe machinery. Consistent offsets would support recalibration; erratic errors would instead point to a poorly generalizing decoder.

```python
print("=== FACTUAL CORRECTION DIAGNOSTICS ===")
base_loss, base_mean = c206_results["original"]
diag = []
for label in ["insulin_only", "combined_weak", "combined_strong"]:
    loss, mean = c206_results[label]
    change = mean - base_mean
    residual = r201_meta.observed.to_numpy() - base_mean
    for e in np.unique(c207_events):
        m = c207_events == e
        diag.append({
            "candidate": label, "event": e, "n": int(m.sum()),
            "original_error_mean": float(residual[m].mean()),
            "correction_mean": float(change[m].mean()),
            "correction_sd": float(change[m].std()),
            "correction_error_cov": float(np.mean(
                (change[m]-change[m].mean())*(residual[m]-residual[m].mean()))),
            "nll_delta": float((loss-base_loss)[m].mean())
        })
print(pd.DataFrame(diag).round(5).to_string(index=False))

print("=== EXISTING DATAFRAME CACHES ===")
for k in list(globals()):
    v = globals()[k]
    if isinstance(v, pd.DataFrame) and any(s in k.lower() for s in ["meta", "panel", "probe"]):
        ids = (v.stay_id.unique() if "stay_id" in v.columns else [])
        print(k, v.shape, "IDs", list(ids[:2]), "columns", list(v.columns)[:16])

print("=== DOSE-PROBE FUNCTION SIGNATURES ===")
for k in ["r85_linear", "r85_norm", "r85_mlp", "r85_bins",
          "r91_lin", "r91_norm", "r91_mlp", "lin96", "mlp96",
          "r129_queries", "r201_ctx", "r201_queries", "r201_fixed"]:
    fn = globals()[k]
    print(k, inspect.signature(fn), "names", fn.__code__.co_names[:24])

print("=== FORWARD OUTPUT INVENTORY ===")
for i in range(1,29):
    path = f"/work/fwd_{i}"
    print(i, os.listdir(path))

print("=== RELEVANT NOTEBOOK EXCERPTS ===")
for run in [90, 96, 115, 138, 145, 164]:
    tag = f"## Run {run}"
    a = _notes.find(tag)
    if a >= 0:
        b = _notes.find("\n## ", a+4)
        print(_notes[a:min(b if b > a else a+550, a+550)])
```
