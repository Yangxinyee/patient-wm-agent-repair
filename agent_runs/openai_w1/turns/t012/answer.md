The architecture confirms that doses affect both the recurrent state and later predictions, while measurement heads also use previous-reading registers. The inspection output was truncated before the reconstruction details, so I’ll examine those separately. This should tell us whether the mismatch comes from state alignment, register encoding, or latent averaging.

```python
import inspect, dis, json
print("=== HEAD / CORE SHAPES ===")
for n in NAMES:
    if n.startswith(("last_", "attr_heads.value", "core.z_", "core.gap", "core.out_norm", "embed.numeric", "embed_transition.value", "query_norm", "type_emb")):
        print(n, SHAPES[n])
print("\n=== VOCABULARY ===")
print("Top-level keys:", list(VOCAB))
for k in ["config", "codes", "normalization", "normalisation"]:
    if k in VOCAB:
        print(k, str(VOCAB[k])[:2500])
print("\n=== RECONSTRUCTION HELPERS ===")
for fn_name in ["activation", "lin", "norm", "mlp", "fine_bin"]:
    fn = globals().get(fn_name)
    if callable(fn):
        print("\n", fn_name)
        try:
            print(inspect.getsource(fn))
        except Exception:
            dis.dis(fn)
print("\n=== RECORD AND STATE ALIGNMENT ===")
print(head_records.columns.tolist())
print(head_records.head(8).to_string(index=False))
print(factual_reference.head(8).to_string(index=False))
print("State keys/shapes:", [(k, np.shape(v)) for k, v in S.items()])
for k in ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]:
    print(k, np.asarray(S[k])[:18])
print("lastnum first rows:", lastnum[:8].cpu().numpy())
print("HM first rows:", HM[:8])
print("Last-related variables:", [(k, type(v).__name__, getattr(v, "shape", None)) 
                                  for k, v in list(globals().items()) if "last" in k.lower()])
print("\n=== SMALL DATA SNAPSHOT ===")
for k in ["train", "df", "data", "tuning"]:
    v = globals().get(k)
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, v.columns.tolist())
        print(v.head(8).to_string(index=False))
```
