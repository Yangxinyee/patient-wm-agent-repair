The local head reconstruction gives NLL near 2.85, whereas the GPU reports about 1.64. That mismatch makes the strength screen—and possibly the centering diagnostics—unreliable. I won’t edit from those results. I’ll check register conventions against saved GPU predictions; agreement is required before trusting local head calculations.

```python
print("=== RECONSTRUCTION NOTES ===")
for _s219 in Path("/work/notes.md").read_text().split("\n## "):
    if any(_s219.startswith(t) for t in [
        "Reconstruction conventions", "Register search",
        "Runs 147", "Runs 153", "Runs 209"
    ]):
        print(("\n## " + _s219)[:2200])

print("=== STATE FILE OBJECTS ===")
for _n219, _v219 in list(globals().items()):
    if isinstance(_v219, np.lib.npyio.NpzFile):
        print(_n219, "file", getattr(getattr(_v219, "fid", None), "name", None),
              "keys", _v219.files)
        for _k219 in ["x", "stay_id", "stay", "hour"]:
            if _k219 in _v219:
                print(" ", _k219, _v219[_k219].shape,
                      _v219[_k219][:3] if _k219 != "x" else "")

print("=== REGISTER AND TYPE TENSORS ===")
for _n219 in ["last_status.weight", "last_numeric.weight",
              "type_emb.weight", "query_norm.weight"]:
    _a219 = BASE217[_n219]
    print(_n219, tuple(_a219.shape),
          "RMS", float(torch.sqrt(torch.mean(_a219 * _a219))))

_logits219 = F.linear(
    _h218, BASE217["attr_heads.value_bin.2.weight"],
    BASE217["attr_heads.value_bin.2.bias"]
) + _tr218
print("local absent-bin probability mean",
      float(torch.softmax(_logits219, -1)[:, 0].mean()))

print("=== FACTUAL FILE SCHEMAS ===")
for _d219 in ["fwd_15", "fwd_16", "fwd_17", "fwd_18", "fwd_19"]:
    for _p219 in sorted(Path("/work", _d219).glob("factual*.csv")):
        _f219 = pd.read_csv(_p219)
        print(str(_p219), _f219.shape, list(_f219.columns))
        print(_f219.head(1).to_string(index=False))

print("=== LAST-STATUS CANDIDATE SCREEN ===")
for _j219 in range(len(BASE217["last_status.weight"])):
    _xx219 = _x218 + BASE217["last_status.weight"][_j219]
    _hh219 = F.gelu(F.linear(
        _xx219, BASE217["attr_heads.value_bin.0.weight"],
        BASE217["attr_heads.value_bin.0.bias"]
    ))
    _ll219 = F.linear(
        _hh219, BASE217["attr_heads.value_bin.2.weight"],
        BASE217["attr_heads.value_bin.2.bias"]
    ) + _tr218
    print("status", _j219,
          "local NLL", round(float(F.cross_entropy(_ll219, _y218)), 6))

with open("/work/notes.md", "a") as _f219:
    _f219.write(
        "\n## Runs 218–219: local-head validation failure\n"
        "- No weights changed. Local head NLL (~2.85) disagrees with GPU factual NLL (~1.64).\n"
        "- Local strength-loss and centering diagnostics are not trusted until register/head reconstruction is validated.\n"
    )
```
