The saved files confirm that the current norepinephrine correction is separate from the insulin baseline. I’ll verify the new regression’s weight mapping before replacing that correction. The check must confirm the admission-null basis and exact edit composition; otherwise the trial could invalidate its local factual safeguards.

```python
import inspect
import numpy as np
from pathlib import Path

print("INPUT-NULL BASIS CANDIDATES")
for name147 in sorted(list(globals())):
    obj147 = globals()[name147]
    if isinstance(obj147, np.ndarray) and obj147.shape == (192, 188):
        print(name147, "orthogonality error",
              float(np.linalg.norm(obj147.T @ obj147 - np.eye(188))))

print("\nRUN 141–142 INTERFACE OBJECTS")
for name147 in sorted(list(globals())):
    if name147.endswith(("141", "142")):
        print(name147, brief146(globals()[name147]))

print("\nREADOUT HELPER")
print("signature", inspect.signature(extract_readout143))
print("global names", extract_readout143.__code__.co_names)
print("output shapes", [brief146(x) for x in base_result143])

original_b147 = np.asarray(load_original("core.b_in.weight")).copy()
current_b147 = np.asarray(load("core.b_in.weight")).copy()
baseline_delta147 = np.load("/work/baseline77.npz")["delta"].copy()
old_ne147 = np.load("/work/ne_trial103.npz")["delta"].copy()
print("\nCOMPOSITION CHECK")
print("baseline norm", float(np.linalg.norm(baseline_delta147)))
print("current minus baseline norm",
      float(np.linalg.norm(current_b147 - original_b147 - baseline_delta147)))
print("current minus baseline minus .35 old NE error",
      float(np.linalg.norm(current_b147 - original_b147
                           - baseline_delta147 - .35 * old_ne147)))
print("output-direction orthogonality error",
      float(np.linalg.norm(Pdir143.T @ Pdir143 - np.eye(14))))

sections147 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("\nLATER IDENTIFICATION / INSULIN VALIDATION")
for label147 in ("Run 63:", "Run 64:", "Run 69:", "Run 70:", "Run 75:", "Run 76:"):
    for section147 in sections147:
        if label147 in section147.split("\n", 1)[0]:
            print(section147[:650])

with open("/work/notes.md", "a") as f147:
    f147.write(
        "\n## Run 147: candidate weight-map verification\n"
        "Audited admission-null basis candidates, output-direction orthogonality, "
        "readout helper interface, and current baseline/NE composition. "
        "No weights changed. Python runs: 147; forwards: 15.\n"
    )
```