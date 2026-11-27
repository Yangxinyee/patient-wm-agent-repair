The original private channels lose almost all prior signal, while the fitted decoder can recover it. Testing the write route against the original projection may therefore have been the wrong check. I’ll test it with the restored controller. Near-unit transmission with little leakage would support this route; failure would rule it out.

```python
print("=== RESTORED-CONTROLLER WRITE TEST ===")
D144 = np.diag(delta143)
print("P.T @ T", np.round(P143.T @ T143, 5))
print("P.T @ D @ T", np.round(P143.T @ D144 @ T143, 5))
print("original write leakage", np.linalg.norm(C143 @ D144 @ T143))
for label, Cnew in [
    ("add P decoder", C143 + R143 @ P143.T),
    ("replace private decoder", C143 + R143 @ (P143-C143.T@R143).T)]:
    response = Cnew @ D144 @ T143
    print(label, "write error", np.linalg.norm(response-R143),
          "private gain", np.round(R143.T @ response, 4),
          "orthogonal leakage", np.linalg.norm(response-R143@(R143.T@response)))
print("minimum-norm decoder inverse error",
      np.linalg.norm(T143 - P143 @ np.linalg.inv(P143.T@P143)))
print("=== CAPTURE DETAILS ===")
for n in ["capture125", "captured_pre118", "pre122", "post122"]:
    v = globals().get(n)
    if isinstance(v, dict):
        print(n, {str(k): (getattr(z, "shape", None), type(z).__name__) for k,z in v.items()})
    else:
        print(n, type(v).__name__, getattr(v, "shape", None))
for name in ["capture_pre125", "linear_capture118"]:
    g = globals()[name]
    text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(g))
    print(name, text[:1300])
print("=== ATTENTION FEATURE RETURN ===")
text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(attention_features116))
print(text[-1800:])
print("=== RECOVERED DATA SHAPES ===")
for n in ["hf_all102", "E_all102", "ctx_all102", "e_valid60", "u_valid60",
          "valid_list103", "estate118", "hp118", "hs118", "hrows118",
          "xf118", "Chead118", "Sstate125", "rhs_state125"]:
    v = globals().get(n)
    print(n, type(v).__name__, getattr(v, "shape", None),
          (len(v) if isinstance(v, list) else ""))
print("=== WIRE ROUTE NORMALISATION ===")
for prefix in ["core.prelude_norm", "core.prelude.0.norm1"]:
    wg = arr142(load_original(prefix+".weight"))
    print(prefix, "SP*weight-QE", np.linalg.norm(arr142(SP125)*wg[:,None]-arr142(QE125)))
print("=== GLOBAL ARRAY NAMES AFTER RUN 125 ===")
print([n for n in globals() if n.endswith(("125", "126", "131", "134", "135", "136", "137", "138")) and
       not n.startswith("_")])
```
