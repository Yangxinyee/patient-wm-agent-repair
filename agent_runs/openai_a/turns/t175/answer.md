The retained notebook contains stronger insulin identification tests, but the output was truncated before their results and the current graft details. I’ll inspect only those sections and the main data objects. This should establish the causal target and the edit’s construction without spending another forward pass.

```python
import types
print("=== TARGETED NOTEBOOK SECTIONS ===")
for i in [12, 13, 21, 22, 29, 31, 36]:
    print(note_sections[i][:1050])
print("=== PANEL ===")
print(panel.shape, list(panel.columns))
print(panel.head(3).to_string(index=False))
print("=== DATAFRAMES ===")
for k, v in sorted(list(globals().items())):
    if type(v).__name__ == "DataFrame":
        print(k, v.shape, list(v.columns)[:8])
print("=== RECONSTRUCTION/HEAD FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if isinstance(v, types.FunctionType) and any(x in k.lower() for x in
        ("head", "register", "poster", "synth", "recon", "step", "graft", "embed")):
        print(k, v.__code__.co_varnames[:8])
print("=== SMALL KEY OBJECTS ===")
for k in ["rd_dynamic_fit", "graft_nodes", "graft_beta", "graft_decoder_b",
          "graft_gate_scale", "graft_alpha", "decode_dose_sets"]:
    v = globals().get(k)
    if isinstance(v, dict):
        print(k, {kk: (tuple(vv.shape) if isinstance(vv, (np.ndarray, torch.Tensor))
                       else type(vv).__name__) for kk, vv in list(v.items())[:12]})
    else:
        print(k, v)
print("VOCAB KEYS", list(VOCAB))
print("CODES", VOCAB["codes"])
```
