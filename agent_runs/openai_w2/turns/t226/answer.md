The architecture confirms that the edit depends on correctly reconstructed bundle features. The new reconstruction is still unverified, despite having the right bundle counts. I’ll check the embedding helpers and state alignment next. A wrong unit lookup or index would explain the discrepancy; consistent inputs and alignment would point to another reconstruction error.

```python
import dis
import numpy as np
import pandas as pd

print("EMBEDDING / RECONSTRUCTION FUNCTION DEPENDENCIES")
for r194_name in (
    "embed_record_atom", "embed_record_atom_v2", "embed_end_repair",
    "bundle_inputs191", "encode_variable_sets_repair", "prelude_features111"
):
    r194_f = globals().get(r194_name)
    if r194_f is not None:
        print(r194_name, "names:", r194_f.__code__.co_names)
        print("constants:", str(r194_f.__code__.co_consts)[:1400])

print("\nDATAFRAMES")
for r194_name, r194_obj in sorted(list(globals().items())):
    if isinstance(r194_obj, pd.DataFrame) and (
        any(s in r194_name.lower() for s in ("meta", "188", "189", "190", "191", "train", "tuning"))
        or r194_obj.shape[0] > 100000
    ):
        print(r194_name, r194_obj.shape, list(r194_obj.columns)[:14])

print("\nSTATE ARCHIVE")
print("keys:", S187.files)
for r194_key in S187.files:
    r194_arr = S187[r194_key]
    print(r194_key, r194_arr.shape, r194_arr.dtype, str(r194_arr[:3])[:300])
print("\nNEW METADATA")
print(meta188.head(8).to_string(index=False))

print("\nRECONSTRUCTION OBJECT NAMES")
print([
    n for n in sorted(globals())
    if any(s in n.lower() for s in (
        "z188", "z189", "z190", "z191", "error191", "old191",
        "old190", "full134", "idx188", "index188", "records188",
        "meta134", "meta111", "unit_ids", "ends191", "bundle191"
    ))
])

print("\nATOM EMBEDDING BYTECODE")
dis.dis(embed_record_atom)

print("\nWEIGHTS PRESERVED:", np.array_equal(
    load("core.b_in.weight"), validated_B176
))
```
