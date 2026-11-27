The architecture confirms that drug inputs affect future readings through the recurrent state, where the rejected edit amplified. The output was truncated before the earlier causal evidence and replay interfaces appeared. I’ll recover those compactly; they are needed to distinguish a supported dynamic effect from a mechanically plausible but unstable edit.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== EARLIER RECORD-BASED EVIDENCE ===")
text133 = Path("/work/notes.md").read_text()
early133 = text133[:-8000]
pars133 = early133.split("\n\n")
selected133 = [
    p for p in pars133
    if any(s in p.lower() for s in [
        "dml", "causal", "kernel", "instrument", "discontinu",
        "confound", "identif", "innovation", "original rollout"
    ])
]
print("Notebook characters:", len(text133))
evidence133 = "\n\n".join(selected133)
print(evidence133[:2400])
if len(evidence133) > 2400:
    print("\n--- Later evidence ---")
    print(evidence133[-1800:])

print("\n=== FUNCTION INTERFACES ===")
for name in [
    "replay_factual125", "transition73",
    "transition_with_b110", "probe_means95"
]:
    fn = globals().get(name)
    if callable(fn):
        print(name, inspect.signature(fn))
        print("locals:", fn.__code__.co_varnames)
        print("dependencies:", fn.__code__.co_names)
        print("defaults:", brief132(fn.__defaults__))
        print("source filename:", fn.__code__.co_filename)

print("\n=== SOURCE / CAUSAL / REPLAY VARIABLE NAMES ===")
names133 = sorted(
    n for n in list(globals())
    if not n.startswith("_")
    and any(s in n.lower() for s in [
        "src", "source", "kernel", "causal", "innovation",
        "replay", "rollout", "target", "tuning", "nll"
    ])
)
print(names133)

print("\n=== CURRENT EDIT MAGNITUDES ===")
for name in edited():
    delta133 = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, "norm", round(float(np.linalg.norm(delta133)), 5),
          "max", round(float(np.max(np.abs(delta133))), 5))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 133\n"
        "Run-132 output was truncated. Compact recovery of causal evidence, "
        "function dependencies and edit magnitudes; no weights changed.\n"
    )
```