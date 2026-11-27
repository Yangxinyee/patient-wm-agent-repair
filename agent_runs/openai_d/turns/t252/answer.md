I found the independent caches and evaluator. The drug edits replace a few hidden neurons; some factual loss may come from losing those neurons’ original contribution rather than from adding dose sensitivity. I’ll separate those costs. If neuron removal is costly, a regression-based compensation could restore factual fit without weakening the drug correction.

```python
r216_orig = c206_candidates["original"]
r216_keys = ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
             "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]
r216_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== CACHE DETAILS ===")
for tag, cache in [("train", c210_train), ("tune100", c210_tune)]:
    print(tag, {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                for k, v in cache.items()})
    print("meta columns", list(cache["meta"].columns))
    print(cache["meta"].head(2).to_string(index=False))
print("=== CHANGED NEURONS ===")
r216_changed = {}
for tag in ["insulin_only", "combined_weak", "combined_strong"]:
    W = c206_candidates[tag]
    rows = np.flatnonzero(
        np.max(np.abs(r216_np(W[r216_keys[0]]) - r216_np(r216_orig[r216_keys[0]])), axis=1) > 1e-6)
    cols = np.flatnonzero(
        np.max(np.abs(r216_np(W[r216_keys[2]]) - r216_np(r216_orig[r216_keys[2]])), axis=0) > 1e-6)
    r216_changed[tag] = np.union1d(rows, cols)
    print(tag, "input rows", rows.tolist(), "output columns", cols.tolist())
print("evaluator names", c210_eval.__code__.co_names)
print("logits names", c211_logits.__code__.co_names)

r216_rows = []
r216_results = {}
for tag, W in [("original", r216_orig),
               ("insulin_only", c206_candidates["insulin_only"]),
               ("combined_strong", c206_candidates["combined_strong"])]:
    for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
        result = c210_eval(W, cache)
        r216_results[(tag, split)] = result
        print("result structure", tag, split,
              [(tuple(x.shape) if hasattr(x, "shape") else type(x).__name__) for x in result])
        r216_rows.append({"candidate": tag, "split": split,
                            "delta": float(np.mean(result[0] - cache["loss0"]))})
for tag in ["insulin_only", "combined_strong"]:
    blank = {k: torch.as_tensor(r216_np(v)).clone() for k, v in r216_orig.items()}
    blank[r216_keys[2]][:, r216_changed[tag]] = 0
    for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
        result = c210_eval(blank, cache)
        r216_results[(tag + "_removed", split)] = result
        r216_rows.append({"candidate": tag + "_removed", "split": split,
                            "delta": float(np.mean(result[0] - cache["loss0"]))})
print("=== LOSS DECOMPOSITION ===")
print(pd.DataFrame(r216_rows).to_string(index=False))

print("=== EARLY ROLLOUT TARGET NOTES ===")
for section in Path("/work/notes.md").read_text().split("\n## "):
    if section.split("\n", 1)[0] == "Run 6":
        print(section[:2200])
        break
```