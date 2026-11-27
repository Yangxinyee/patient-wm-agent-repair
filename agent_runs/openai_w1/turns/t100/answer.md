The architecture confirms that dose changes enter both the action encoder and the latent posterior, then propagate through recurrent state. The retained trace candidates are linear fits, but their intended targets and validation results still need checking.

I’ll recover that evidence before choosing an edit. The key test is whether the proposed trace represents drug exposure—not merely the patient condition that prompted treatment.

```python
from pathlib import Path
import numpy as np

r91_notes = Path("/work/notes.md").read_text()
print("=== EARLY EVIDENCE AND ORIGINAL ROLLOUTS ===")
print(r91_notes[:10500])

print("\n=== CANDIDATE DEVELOPMENT ===")
r91_start = r91_notes.find("## Run 45")
if r91_start < 0:
    r91_start = r91_notes.find("## Runs 45")
if r91_start >= 0:
    r91_stop = r91_notes.find("## Run 60", r91_start)
    print(r91_notes[r91_start:r91_stop if r91_stop >= 0 else r91_start+11000][:11000])

print("\n=== DATA DICTIONARY ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED OBJECTS: REPAIR, TRACE, POSTERIOR ===")
for r91_name, r91_obj in sorted(list(globals().items())):
    if r91_name.startswith(("repair_", "rr51_", "rr58_")) and not callable(r91_obj):
        print(r91_name, type(r91_obj).__name__, getattr(r91_obj, "shape", ""))
        if isinstance(r91_obj, dict):
            print(" keys:", [str(k)[:70] for k in list(r91_obj)[:18]])

print("\n=== TRACE FIT SUMMARY ===")
for r91_key, r91_candidate in rr51_trace_candidates.items():
    print("ridge", r91_key, "score", r91_candidate[0],
          "coefficient norms", np.linalg.norm(r91_candidate[1], axis=1).tolist(),
          "bias", np.asarray(r91_candidate[2]).tolist())
for r91_key, r91_assignment in rr51_assignment.items():
    print("assignment", r91_key, "intercept/score", r91_assignment[1:],
          "coefficient norm", float(np.linalg.norm(r91_assignment[0])))

with open("/work/notes.md", "a") as r91_f:
    r91_f.write(
        "\n## Run 91\n"
        "Architecture recovered: action encoder and z posterior both mediate dose effects. "
        "Recovering original causal evidence and trace targets before selecting weight surgery. "
        "No tensors edited. Counts: 91 Python runs, 6 forward passes.\n"
    )
```
