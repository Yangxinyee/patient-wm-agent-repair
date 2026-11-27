The retained cache contains factual logits, observed bins and sensor identities, so I can fit a small categorical calibration from training records without gradients through the model. I’ll test sensor-specific bin offsets offline first. Lower loss on both validation sets would support this approach; improvement confined to the fitting rows would refute it.

```python
def arr279(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

Torig279 = np.vstack([
    arr279(load_original("attr_heads.value_bin.2.weight")).T,
    arr279(load_original("attr_heads.value_bin.2.bias"))[None, :]
]).astype(np.float64)
Tcurrent279 = np.vstack([
    arr279(load("attr_heads.value_bin.2.weight")).T,
    arr279(load("attr_heads.value_bin.2.bias"))[None, :]
]).astype(np.float64)

offset_train279 = L179 - X179 @ Torig279
logits_train279 = arch263["F"] @ Tcurrent279 + offset_train279
logits_tune279 = stats263["Ftune"] @ Tcurrent279 + offset234
ntrain279 = np.asarray(nll179(logits_train279))
ntune279 = np.asarray(nll228(logits_tune279))
print("NLL array shapes", ntrain279.shape, ntune279.shape)
print("Baseline deltas",
      float((ntrain279-base_nll_train234)[fitmask179].mean()),
      float((ntrain279-base_nll_train234)[valmask179].mean()),
      float((ntune279-base_nll_tune234).mean()))
print("Stored score", grid269[(.25,.25)]["score"])
print("Final-head reconstruction error",
      float(np.max(np.abs(Tcurrent279-grid269[(.25,.25)]["T"]))))

events279 = np.asarray(events179)
codes279 = {}
for e279 in np.unique(events279):
    v279 = code_lookup254[e279]
    if isinstance(v279, dict):
        v279 = next(v279[k] for k in ("id", "index", "idx", "code_id") if k in v279)
    codes279[e279] = int(v279)
print("Code IDs", codes279)
tune_codes279 = ix_tune253 // 17
probs_train279 = softmax179(logits_train279, axis=-1).mean(axis=0)
print("Observed-bin agreement", np.array_equal(B179, observed_bins82))

calgrid279 = {}
for alpha279 in (25., 100., 400., 1600.):
    offsets279 = {}
    for e279 in codes279:
        use279 = fitmask179 & (events279 == e279)
        expected279 = probs_train279[use279].sum(axis=0)
        observed279 = np.bincount(B179[use279], minlength=17).astype(float)
        pseudo279 = alpha279 * expected279 / max(int(use279.sum()), 1)
        d279 = np.log((observed279 + pseudo279 + 1e-8) /
                      (expected279 + pseudo279 + 1e-8))
        d279[0] = 0.
        d279[1:] -= np.average(d279[1:], weights=expected279[1:]+1e-8)
        offsets279[e279] = d279
    for strength279 in (.2, .5, 1.):
        dtrain279 = np.zeros((len(events279), 17))
        dtune279 = np.zeros((len(tune_codes279), 17))
        for e279, code279 in codes279.items():
            dtrain279[events279 == e279] = strength279 * offsets279[e279]
            dtune279[tune_codes279 == code279] = strength279 * offsets279[e279]
        nc279 = np.asarray(nll179(logits_train279 + dtrain279[None]))
        nv279 = np.asarray(nll228(logits_tune279 + dtune279[None]))
        score279 = (
            float((nc279-base_nll_train234)[fitmask179].mean()),
            float((nc279-base_nll_train234)[valmask179].mean()),
            float((nv279-base_nll_tune234).mean())
        )
        calgrid279[(alpha279,strength279)] = {
            "offsets": {e: strength279*d for e,d in offsets279.items()},
            "score": score279,
            "relative": (float((nc279-ntrain279)[valmask179].mean()),
                         float((nv279-ntune279).mean()))
        }

print("=== CALIBRATION CANDIDATES: factual deltas; validation gains ===")
for key279, item279 in sorted(
    calgrid279.items(), key=lambda kv: max(kv[1]["relative"])
):
    print(key279, np.round(item279["score"], 6),
          "relative", np.round(item279["relative"], 6))

with open("/work/notes.md", "a") as f279:
    f279.write("\n## Run 279 / records-fitted categorical calibration\n"
               "Fit closed-form sensor-specific bin-count offsets on training "
               "fitmask only, with shrinkage toward model probabilities. "
               "Compared training-validation and independent tuning factual NLL. "
               "No model weights saved; candidates retained in calgrid279.\n")
```