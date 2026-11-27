The new states cover 100 independent stays and 4,144 bundles. They do not yet establish a better edit. I’ll check how to reconstruct their prelude features and verify the encoder against the existing cohort before fitting anything. Bundle counts and posterior-state agreement should expose encoding mistakes.

```python
import inspect
import dis
import pandas as pd
from pathlib import Path

print("RECONSTRUCTION HELPERS")
for name in ["previous_context_repair", "embed_record_atom",
             "embed_record_atom_v2", "batched_set_repair"]:
    fn = globals()[name]
    print("\n", name, inspect.signature(fn))
    print("Globals:", fn.__code__.co_names)
    try:
        print(inspect.getsource(fn))
    except (OSError, TypeError):
        if name in ["previous_context_repair", "embed_record_atom_v2"]:
            dis.dis(fn)

print("\nCODE AND UNIT MAPS")
for name in ["unit_ids_repair", "rate_unit_ids_repair"]:
    print(name, globals()[name])
print("VOCAB keys:", list(VOCAB.keys()))

print("\nAVAILABLE RECORD TABLES")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and (
        "stay_id" in obj.columns and
        ("event" in obj.columns or name.endswith("repair"))
    ):
        print(name, obj.shape, list(obj.columns))

print("\nREFERENCE BUNDLES")
print(sm_repair.head(12).to_string(index=True))
print(reference_records_repair.head(14).to_string(index=False))
print("First 12 observation/action atom counts:",
      [(len(obs_sets_repair[i]), len(action_records_repair[i]))
       for i in range(12)])

notes187 = Path("/work/notes.md").read_text()
lines187 = notes187.splitlines()
print("\nFEATURE-RECONSTRUCTION NOTE EXCERPTS")
for i, line in enumerate(lines187):
    if any(key in line.lower() for key in
           ["end route", "end atom", "posterior reconstruction",
            "prelude reconstruction", "run 100", "run100",
            "run 111", "run111", "run 134", "run134"]):
        print("\n".join(lines187[max(0, i-1):min(len(lines187), i+4)]))

S187 = np.load("/work/fwd_24/states_original.npz")
print("\nNEW STATE FILE:", S187["x"].shape,
      "unique stays:", len(np.unique(S187["stay_id"])))
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward24 / run187: independent states\n"
            "Obtained original states for 100 independent training stays (4144 bundles). "
            "Current weights unchanged. Reconstruct and validate features before refitting.\n")
```