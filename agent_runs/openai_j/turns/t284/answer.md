The saved rollout targets are about −10.53/+5.26 for insulin and +3.81/−1.90 for norepinephrine on the first 100 stays. These remain assumption-dependent estimates, especially given insulin’s lag-sensitive IV results. I’ll recover only the essential helper details and validation IDs, without printing large function defaults.

```python
import inspect
from pathlib import Path
import numpy as np

print("=== LAST TUNING BATCH ===")
print([int(x) for x in tuning_ids249[300:400]])

print("\n=== RELEVANT HELPERS: NAMES ONLY ===")
print([k for k, v in sorted(list(globals().items()))
       if not k.startswith("_") and inspect.isfunction(v)
       and v.__module__ == "__main__"
       and any(t in k.lower() for t in
               ["make", "score", "head", "eval", "install", "kernel", "contrast"])])

print("\n=== MAKE242 DEPENDENCIES ===")
print(make242.__code__.co_names)
try:
    print(inspect.getsource(make242)[:2500])
except Exception:
    import dis
    dis.dis(make242)

print("\n=== RELEVANT LIVE OBJECTS ===")
deps250 = set(make242.__code__.co_names)
for k, v in sorted(list(globals().items())):
    if k in deps250 or k in ["safer247", "candidate247", "candidate243", "head177_eval"]:
        if isinstance(v, np.ndarray):
            print(k, "array", v.shape)
        elif isinstance(v, dict):
            print(k, "dict", [(a, getattr(b, "shape", type(b).__name__))
                            for a, b in list(v.items())[:10]])
        elif isinstance(v, (int, float, str)):
            print(k, v)
        elif callable(v):
            print(k, "callable")

print("\n=== IDENTIFICATION NOTE HEADINGS ===")
print([(i, s.split("\n")[0]) for i, s in enumerate(sections249)
       if any(t in s.split("\n")[0].lower()
              for t in ["kernel", "instrument", "causal", "identif", "iv"])])

print("\n=== LATEST KERNEL / IDENTIFICATION TEXT ===")
evidence250 = [s for s in sections249
               if any(t in s.split("\n")[0].lower()
                      for t in ["kernel", "instrument", "identif"])]
print("\n".join(s[:800] for s in evidence250[-3:]))

print("\n=== CURRENT CHECKPOINT KEYS ===")
with np.load("/work/safer_candidate247.npz") as ck250:
    print([(k, ck250[k].shape) for k in ck250.files])

with Path("/work/notes.md").open("a") as fh250:
    fh250.write(
        "\n## Run 250: bounded-output recovery\n"
        "- Record-based first-100 kernel targets: insulin -10.5288/+5.2644;"
        " NE +3.8084/-1.9042, conditional on identification assumptions.\n"
        "- Insulin dynamic-IV estimates are lag-sensitive; do not mistake the targets"
        " for directly observed counterfactual truth.\n"
        "- No weights changed; recovering construction and independent validation IDs.\n"
    )
```