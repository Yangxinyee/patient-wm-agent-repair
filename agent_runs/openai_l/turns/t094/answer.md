The target function aims for a one-step response of −1.95 for insulin and +3.1 for norepinephrine, but the diagnostic output was truncated before the supporting evidence appeared. I’ll retrieve that evidence and the original rollout results compactly. Those results must support both the targets and their timing before I use them in a weight edit.

```python
from pathlib import Path
import re, inspect

print("=== RELEVANT WEIGHTS ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_prior", "z_in", "z_head", "attr_heads.value", "encode_action")):
        print(name, SHAPES[name])

print("\n=== FUNCTION DEPENDENCIES ===")
for name in ["make_cf81", "evaluate83", "decode67", "transition64",
             "context64", "bundle_inputs74", "compact_bytecode71"]:
    fn = globals()[name]
    print(name, inspect.signature(fn))
    print(" locals:", fn.__code__.co_varnames)
    print(" globals:", fn.__code__.co_names)
    print(" constants:", repr(fn.__code__.co_consts)[:1000])

print("\n=== CAUSAL EVIDENCE NOTES ===")
notes86 = Path("/work/notes.md").read_text()
sections86 = re.split(r"(?=^## )", notes86, flags=re.M)
for section in sections86:
    if re.match(r"## (Run|Runs) (1[3-9]|2[0-4])\b", section):
        print(section[:850])

print("\n=== ORIGINAL DOSE ROLLOUTS ===")
for p in sorted(Path("/work").glob("fwd_*/dose_edit*.csv")):
    t86 = pd.read_csv(p)
    print(str(p), t86.shape, t86.columns.tolist())
    print(t86.head(4).to_string(index=False))

print("\n=== LATEST FIT NOTES ===")
for section in sections86:
    if re.match(r"## (Run|Runs) 8[0-3]\b", section):
        print(section[:1100])
```