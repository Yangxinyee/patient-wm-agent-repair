The saved decoder was tested over broad dose ranges, and some errors were much larger than its average RMSE. I’ll check the normalization correction and inspect errors by treatment configuration. Small errors on recorded doses but failures on doubled or tripled doses would make this decoder unsuitable for the repair.

```python
import dis
from collections import defaultdict

def compact_function109(f):
    groups = defaultdict(list)
    line = f.__code__.co_firstlineno
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname not in {"RESUME", "CACHE", "NOP", "END_FOR"}:
            groups[line].append(ins.opname + (" " + ins.argrepr if ins.argrepr else ""))
    print(f.__name__, str(inspect.signature(f)))
    for line, ops in groups.items():
        print(line, " | ".join(ops))

for n in ["action_hidden105", "channel_targets105", "wm_lin", "wm_norm",
          "wm_set", "wm_layer"]:
    print("\n=== ", n, " ===")
    compact_function109(globals()[n])

print("\n=== SELECTED DECODER ERRORS BY CONFIGURATION ===")
decoder109 = best105[1]
pred109 = decoder109.predict(xst105) * xsd105[:0].sum() if False else None
print("rate scales", rate_scale105, "caps", caps105, "delta", delta105)
print("normalization metadata", repr(norm_meta)[:1300])
print("test rate ranges", np.min(qtest105, axis=0), np.max(qtest105, axis=0))
print("test presence counts", np.unique(ptest105, axis=0, return_counts=True))
print("Ridge feature/target scaling objects:")
for n in ["xmean105", "xsd105", "base_var_test105", "yt105"]:
    x = globals()[n]
    print(n, "range", float(np.nanmin(x)), float(np.nanmax(x)))

print("\n=== ATTENTION CONFIGURATION ===")
print(VOCAB["config"])
print("\n=== CORE AND EMBEDDING TENSORS ===")
for n in NAMES:
    if n.startswith(("core.", "embed.", "type_emb.", "encode.norm", "encode_action.norm")):
        if not any(s in n for s in ("blocks.", "prelude.", "coda.", "z_prior.", "z_post.", "age.")):
            print(n, SHAPES[n])
```
