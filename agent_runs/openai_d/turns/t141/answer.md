The inventory confirms that the cached head inputs cover 4,510 training readings, but the scan stopped on an unrelated object’s `shape` attribute. No weights changed. I’ll load the state file directly and verify its alignment with the head cache before making a controlled probe; disagreement would mean the contrast cannot yet be trusted.

```python
r123_statefiles = sorted(Path("/work").glob("fwd_*/states_original.npz"))
print("=== STATE FILES ===")
for p in r123_statefiles:
    with np.load(p) as s:
        print(str(p), {k: s[k].shape for k in s.files})
        if s["x"].shape[0] == 4035:
            r123_states = {k: s[k].copy() for k in s.files}
            r123_statepath = str(p)

print("=== EXACT HEAD TENSOR PATHS ===")
for n in NAMES:
    if any(s in n for s in ["gap", "type_emb", "last_", "query_norm", "out_norm"]):
        print(n, SHAPES[n])
print("=== HELPERS ===")
dis.dis(norm96)
dis.dis(act96)
print("=== CONSTRUCTION OBJECTS ===")
for n in ["theta104", "tilt104", "baseL104", "removed104",
          "hbar104", "v104", "b104", "gate_strength104", "fits99", "gates99"]:
    if n in globals():
        ob = globals()[n]
        sh = getattr(ob, "shape", None)
        print(n, type(ob).__name__, sh if isinstance(sh, (tuple, torch.Size)) else "")
        if isinstance(ob, dict):
            print(str(ob)[:1300])
        elif np.isscalar(ob) or n == "hbar104":
            print(ob)
print("=== EMBEDDING NORMALIZATIONS ===")
print("cached base/context max difference", (base97 - ctx98).abs().max().item())
print("fixed vs head-q max difference",
      (xx98[0] - qq98[0] - fixed96).abs().max().item())
print("=== RECORD SAMPLE ===")
print(train.loc[train.stay_id == 1010000002].head(15).to_string(index=False))
```
