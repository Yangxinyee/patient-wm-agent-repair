The architecture shows that doses affect a shared recurrent state, while measurement predictions also use last-value registers. A value-head-only edit may spare the other outputs, but it must separate dose effects from treatment selection. I’ll inspect the residualization results and relevant tensors next; consistent effects across sensors and patient subsets would support proceeding.

```python
print("=== ANALYSIS VARIABLES AFTER Last ===")
gv = list(globals().items())
start = next((i for i, (k, v) in enumerate(gv) if k == "Last"), len(gv))
for k, v in gv[start:]:
    if k.startswith("_") or type(v).__name__ in {"module", "function", "type"}:
        continue
    print(k, type(v).__name__, getattr(v, "shape", ""),
          ("keys=" + str(list(v.keys())[:12])) if isinstance(v, dict) else "")

print("\n=== RECENT FIT SUMMARIES ===")
for k, v in gv[start:]:
    if isinstance(v, dict) and any(s in k.lower() for s in ["fit", "result", "estimate", "dml", "cross"]):
        print(k)
        for kk, vv in list(v.items())[:20]:
            text = repr(vv)
            print(kk, text[:650])

print("\n=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(n.startswith(p) for p in [
        "attr_heads.value_bin", "last_numeric", "embed.numeric",
        "embed.bins", "embed_transition.value_bin", "encode_action",
        "query_norm", "type_emb", "embed.code", "core.out_norm"
    ]):
        print(n, SHAPES[n])

print("\n=== STATES ===")
s3 = np.load("/work/fwd_3/states_original.npz")
for k in s3.files:
    print(k, s3[k].shape, s3[k].dtype)
print("\n=== SCHEMA ===")
print(open("/data/README.md").read())
```
