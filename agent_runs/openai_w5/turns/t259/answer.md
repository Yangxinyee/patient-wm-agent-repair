The saved states and reading rows are available, but the diagnostic output was again truncated before the useful details. I’ll compact the helper inspection and check the register dimensions and timing explicitly. This should reveal any indexing error without another lengthy replay attempt.

```python
import dis
import numpy as np
from collections import defaultdict

print("=== HEAD225 OPERATIONS BY SOURCE LINE ===")
groups230 = defaultdict(list)
line230 = None
keep230 = {"LOAD_CONST", "LOAD_GLOBAL", "LOAD_FAST", "LOAD_ATTR", "LOAD_METHOD",
           "STORE_FAST", "BINARY_OP", "UNARY_NEGATIVE", "COMPARE_OP"}
for ins in dis.get_instructions(head225):
    if ins.starts_line is not None:
        line230 = ins.starts_line
    if ins.opname in keep230:
        groups230[line230].append(ins.opname.replace("LOAD_", "").replace("STORE_", "SET_")
                                 + ":" + ins.argrepr[:80])
for line, ops in groups230.items():
    print(line, " ".join(ops))

print("\n=== HEAD TENSOR DIMENSIONS ===")
for name in NAMES:
    if any(name.startswith(p) for p in
           ("last_numeric", "last_status", "type_emb", "query_norm", "z_prior", "gap", "attr_heads.value_bin")):
        print(name, SHAPES[name])

print("\n=== ALIGNMENT AND REGISTER SAMPLE ===")
for name in ["S", "sf225", "state_file"]:
    obj = globals().get(name)
    if isinstance(obj, np.lib.npyio.NpzFile):
        print(name, obj.zip.filename)
print(head_rows[["stay_id", "hour", "event", "value", "last_value", "last_hour"]].head(8).to_string(index=False))
print("head_prev", head_prev[:8], "last bins", head_vb[:8])
print("last_numeric inputs", head_ln[:4].tolist())
print("GPU sample", gpu225.head(3).to_dict("records"))
print("state sample",
      [(int(S["stay_id"][i]), float(S["hour"][i]), int(S["n_atoms"][i])) for i in range(10)])
print("numeric glucose schema", num_spec[("glucose_finger", "value")])
print("type indices / shapes", VOCAB["config"], VOCAB["atom_kinds"], VOCAB["event_kinds"])
```
