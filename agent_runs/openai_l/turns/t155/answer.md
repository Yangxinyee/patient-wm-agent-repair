The fitted decoder recovers flowed private states with about 0.012 RMS error, and the dose channels retain useful contrasts. Their factual variation is not negligible, so factual fit still needs an explicit audit. I’ll identify the exact low-rank controller edit next; it must preserve ordinary predictions while restoring private-state persistence.

```python
C143 = arr142(load_original("core.c_out.weight"))
B143 = arr142(load_original("core.b_in.weight"))
R143 = arr142(R118)
P143 = arr142(P125)
T143 = arr142(T125)
print("=== CONTROLLER IDENTITIES ===")
for label, a in [
    ("C.T @ R", C143.T @ R143),
    ("P - C.T @ R", P143 - C143.T @ R143),
    ("P / 10", P143 / 10)]:
    print(label, "norm", np.linalg.norm(a), "error to T", np.linalg.norm(a-T143))
print("original persistence",
      np.round((X142 @ (C143.T @ R143))[:4], 4))
print("C/B norms", np.linalg.norm(C143), np.linalg.norm(B143))
s143 = np.linalg.svd(C143, compute_uv=False)
print("C singular range", s143.min(), s143.max())
delta143 = np.logaddexp(0., arr142(load_original("core.delta_in")))
print("delta range/mean", delta143.min(), delta143.max(), delta143.mean())
print("=== NORMALISATION IDENTITY ===")
for prefix in ["core.prelude_norm", "core.prelude.0.norm1", "core.prelude.0.norm2"]:
    print(prefix,
          np.linalg.norm(arr142(load_original(prefix+".weight"))-arr142(gn125)),
          np.linalg.norm(arr142(load_original(prefix+".bias"))-arr142(bn125)))
print("SP*gn minus QE norm", np.linalg.norm(arr142(SP125)*arr142(gn125)[:,None]-arr142(QE125)))
print("=== CAPTURE DETAILS ===")
for n in ["capture125", "captured_pre118", "pre122", "post122"]:
    v = globals().get(n)
    if isinstance(v, dict):
        print(n, {k: (getattr(z, "shape", None), type(z).__name__) for k,z in v.items()})
    else:
        print(n, type(v).__name__, getattr(v, "shape", None))
for name in ["capture_pre125", "linear_capture118"]:
    g = globals()[name]
    text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(g))
    print(name, text[:1600])
print("=== ATTENTION FEATURE RETURN ===")
g = attention_features116
text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(g))
print(text[-2100:])
print("=== RELEVANT RECOVERED DATA ===")
for n in ["hf_all102", "E_all102", "ctx_all102", "e_valid60", "u_valid60",
          "valid_list103", "estate118", "hp118", "hs118", "hrows118",
          "coef113", "deltas113", "Chead118", "xf118"]:
    v = globals().get(n)
    print(n, type(v).__name__, getattr(v, "shape", None),
          (len(v) if isinstance(v, list) else ""))
```
