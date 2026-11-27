The architecture confirms that doses act through a nonlinear state update, so scaling a latent column need not scale the drug response. The output was truncated before the targets and helper details appeared. I’ll retrieve those compactly; an accurate reconstruction of official predictions remains the test for trusting local calibration.

```python
import json, inspect, dis
from pathlib import Path

print("=== COMPACT OFFICIAL HISTORY ===")
for k215 in range(1, 15):
    p215 = Path(f"/work/fwd_{k215}/summary.json")
    if not p215.exists():
        continue
    s215 = json.loads(p215.read_text())
    row215 = {"pass": k215, "type": s215.get("type")}
    for model215, result215 in s215.get("results", {}).items():
        if "edits" in result215:
            row215[model215] = {
                key215: val215.get("mean_shift")
                for key215, val215 in result215["edits"].items()
            }
        elif "value_nll_per_value" in result215:
            row215[model215] = result215["value_nll_per_value"]
    print(row215)

print("\n=== EARLY IDENTIFICATION NOTES ===")
lines215 = Path("/work/notes.md").read_text().splitlines()
hits215 = [i215 for i215, line215 in enumerate(lines215)
           if any(key215 in line215.lower()
                  for key215 in ["causal", "instrument", "impulse", "confound",
                                 "identif", "true effect", "retention"])]
for i215 in hits215[:24]:
    print(i215, lines215[i215])

print("\n=== HEAD IMPLEMENTATION CANDIDATES ===")
candidates215 = []
for name215, obj215 in list(globals().items()):
    if not callable(obj215) or not hasattr(obj215, "__code__"):
        continue
    code215 = obj215.__code__
    constants215 = " ".join(x215 for x215 in code215.co_consts
                           if isinstance(x215, str))
    if ("last_numeric" in constants215 or "attr_heads.value_bin" in constants215
            or "head" in name215.lower()):
        candidates215.append(name215)
        print(name215, "args", code215.co_varnames[:code215.co_argcount],
              "uses", code215.co_names,
              "keys", constants215[:350])

print("\n=== CAPTURE HELPER SOURCE ===")
for name215 in ["r179_capture_head", "r186_heads", "r186_means"]:
    obj215 = globals().get(name215)
    if obj215 is None:
        continue
    print(name215)
    try:
        print(inspect.getsource(obj215)[:2400])
    except Exception:
        instructions215 = list(dis.Bytecode(obj215))
        print("\n".join(f"{x215.opname} {x215.argrepr}"
                        for x215 in instructions215[:65]))

print("\n=== DATA AND VOCAB KEYS ===")
print("VOCAB keys", list(VOCAB))
print(pd.read_csv("/data/train.csv", nrows=12).to_string(index=False))

with open("/work/notes.md", "a") as f215:
    f215.write("\n## Run 215: compact recovery after truncated output\n"
               "- Architecture confirms nonlinear, parallel action/posterior routing.\n"
               "- Recovering only official scalar results, causal evidence, and head implementation metadata.\n"
               "- No weights changed. Python runs: 215; forward passes: 14.\n")
```