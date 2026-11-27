The pipeline is intact, but its decoder used states from only 100 training stays. The latest factual cohort exposes a generalization risk, so I will not increase the gains. I’ll recheck the construction before adding independent constraints. A refinement must preserve dose sensitivity while reducing unseen-stay factual drift.

```python
from pathlib import Path
import inspect
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(Path("/data/README.md").read_text())

print("\nSAVED EDIT OBJECTS")
for name in ["validated_B151", "validated_B176", "validated_W151",
             "validated_W176", "validated_gains151", "V128",
             "mu147", "sd147", "C147", "penalty171"]:
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", None))
print("Edited tensors:", edited())
print("Matches validated_B176:",
      np.array_equal(np.asarray(load("core.b_in.weight")),
                     np.asarray(validated_B176)))

print("\nHELPER SIGNATURES")
for name in ["previous_context_repair", "prelude_features111",
             "embed_record_atom", "embed_record_atom_v2",
             "encode_variable_sets_repair", "encode_atom_set",
             "fold_decoder147", "fit_zero_fact171"]:
    if name in globals():
        fn = globals()[name]
        print(name, inspect.signature(fn))
        print("  constants:", str(fn.__code__.co_consts)[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run186: restart recovery\n"
            "Current edit preserved. Rechecking architecture and decoder construction "
            "before adding independent training-state constraints; do not strengthen "
            "gains given the third-cohort factual-budget failure.\n")
```