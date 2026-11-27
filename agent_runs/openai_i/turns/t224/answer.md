The inspection stopped because it treated the tensor-shape table as a weight dictionary. That was an inspection error; no weights changed and no channel test ran. I’ll exclude metadata and repeat the focused checks. The key question remains whether dose innovations can be routed with little factual leakage.

```python
print("=== ACTUAL POSTERIOR WEIGHT DICTIONARIES ===")
for n, v in list(globals().items()):
    if not isinstance(v, dict) or "z_post.0.weight" not in v:
        continue
    if not isinstance(v["z_post.0.weight"], (np.ndarray, torch.Tensor)):
        continue
    print(n, "keys:", len(v))
    for k in ("z_post.0.weight", "z_post.2.weight"):
        if k in v:
            a = array196(v[k])
            if a.shape == tuple(SHAPES[k]):
                print(" ", k, "RMS delta",
                      float(np.sqrt(np.mean((a-array196(load_original(k)))**2))))

print("\n=== CHANNEL OBJECTS ===")
for n, v in list(globals().items()):
    if n.startswith("_"):
        continue
    if any(s in n for s in ("191", "192", "193", "195")):
        if isinstance(v, dict):
            print(n, "dict", list(v)[:5])
        elif callable(v):
            print(n, "function", v.__code__.co_names)
        elif hasattr(v, "shape"):
            print(n, tuple(v.shape))

print("\n=== GATED INNOVATION STATISTICS ===")
print("private dims:", private_dims196, "head:", private_head196)
print("F195 quantiles:", np.quantile(F195, [.01, .1, .5, .9, .99], axis=0))
print("Posterior/private coordinate RMS:",
      np.sqrt(np.mean(Zbase196[:, private_dims196]**2, axis=0)))
print("z_in private-column norms:",
      np.linalg.norm(zin_newW196[:, private_dims196], axis=0))
print("z_in bias norm:", np.linalg.norm(zin_newb196),
      "anchor scale:", anchor_scale196)
print("Prelude decoder diagnostics:")
print(pd.DataFrame(rows194).head(6).to_string(index=False))

print("\n=== HELPER CONTRACTS ===")
for n in ("norm136", "linear136", "lin136", "core_step136", "posterior137",
          "prelude191", "prelude192", "attention_features196"):
    fn = globals().get(n)
    if callable(fn):
        print(n, inspect.signature(fn), "names:", fn.__code__.co_names)
        print("constants:", fn.__code__.co_consts)

print("\n=== LAG EVIDENCE NOTES ===")
_count200 = 0
for i, s in enumerate(_lines197):
    if re.search(r"impulse|persistence|lag.?[0-9]|target.*shift|record.*rollout|\bAR\b", s, re.I):
        print("\n".join(_lines197[max(0, i-1):min(len(_lines197), i+3)]))
        _count200 += 1
        if _count200 >= 12:
            break
print("\nActive edits:", edited())
```
