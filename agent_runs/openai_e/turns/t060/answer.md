The architecture confirms that the current edit changes recurrent state updates, so its factual leakage can accumulate during rollouts. I’ll recover the record-based effect estimates and inspect the value-head structure before choosing between a more selective state edit and factual calibration. Calibration would help only if the trial’s factual errors are systematic, rather than unpredictable distortion.

```python
from pathlib import Path
import inspect
import re

print("DATA DESCRIPTION")
print(Path("/data/README.md").read_text())

print("\nEARLY EVIDENCE FROM NOTEBOOK")
notes54 = Path("/work/notes.md").read_text()
early54 = notes54.split("## Run 33")[0]
lines54 = early54.splitlines()
selected54 = [
    line for line in lines54
    if re.search(
        r"beta|effect|persistence|rho|instrument|confound|policy|NLL|causal|identif",
        line, re.I
    )
]
print("\n".join(selected54)[:4800])

print("\nHELPER NAMES")
helpers54 = []
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            helpers54.append(f"{name}{inspect.signature(obj)}")
        except Exception:
            helpers54.append(name)
print("\n".join(sorted(helpers54))[:2200])

print("\nRELEVANT TENSOR SHAPES")
for name in NAMES:
    if any(s in name for s in (
        "attr_heads.value", "embed_transition.value", "last_numeric",
        "last_status", "query_norm", "type_emb", "core.b_in", "z_head"
    )):
        print(name, SHAPES[name])

print("\nRELEVANT RETAINED NAMES")
print(sorted(
    name for name in globals()
    if not name.startswith("_") and any(s in name.lower() for s in (
        "factual", "rollout", "trial", "beta", "policy", "chosen",
        "coefficient", "fit47", "metrics47"
    ))
))
print("\nTRIAL FIT METRICS")
if "slot_metrics47" in globals():
    print(slot_metrics47)
if "dose_policy" in globals():
    for key, value in dose_policy.items():
        print("dose_policy", key, type(value).__name__)
        if isinstance(value, dict):
            print("keys:", list(value.keys()))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 54: causal evidence and factual-calibration inventory\n"
        "Architecture confirms the current b_in edit modifies recurrent updates. "
        "Recovering record-based causal estimates, value-head tensor structure, "
        "and retained helpers to distinguish systematic factual bias from "
        "state distortion. No weights changed. Python runs: 54; forwards: 5.\n"
    )
```