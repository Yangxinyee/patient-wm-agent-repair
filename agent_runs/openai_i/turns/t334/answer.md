The recovered notes show that raw dose–reading associations reverse after accounting for prior condition, so they cannot be used as drug effects directly. The printout was truncated before the final estimates and stay IDs. I’ll extract those compactly, then compare the candidate’s rollouts with the record-based targets.

```python
import re, json, numpy as np

print("=== IDENTIFICATION HEADINGS ===")
headings288 = [
    s for s in lines287
    if s.startswith("##") and
    re.search(r"identif|causal|effect|innovation|dynamics|saturation", s, re.I)
]
print("\n".join(headings288[:22]))

print("=== NUMERIC TARGET NOTES ===")
targetlines288 = [
    s for s in lines287
    if re.search(r"target|truth|record-derived|true effect|causal estimate", s, re.I)
    and re.search(r"\d", s)
]
chosen288 = targetlines288[:26] + targetlines288[-10:]
print("\n".join(dict.fromkeys(chosen288)))

print("=== LATEST FACTUAL COHORT ===")
print(json.dumps([int(x) if isinstance(x, (int, np.integer)) else x for x in cohort287]))

print("=== SAVED CANDIDATE KEYS ===")
for stem288 in ["neutral209", "corrected269", "token257", "corrected_direct286"]:
    with np.load("/work/" + stem288 + ".npz", allow_pickle=False) as z288:
        print(stem288, [(k, z288[k].shape) for k in z288.files])

print("=== RELEVANT LIVE HELPERS ===")
print(sorted(k for k in list(globals())
             if not k.startswith("_") and callable(globals()[k])
             and re.search(r"replay|install|response|token|posterior|action|direction|mean|fit|bundle|route|gate", k, re.I)))

print("=== RELEVANT LIVE ARRAYS / OBJECTS ===")
print([(k, getattr(globals()[k], "shape", type(globals()[k]).__name__))
       for k in sorted(list(globals()))
       if not k.startswith("_") and not callable(globals()[k])
       and re.search(r"target|effect|mean_direction|response_direction|beta|gain|gate", k, re.I)][-45:])

with open("/work/notes.md", "a") as f288:
    f288.write(
        "\n## Python 288 — compact restart recovery\n"
        "- Prior output was truncated; extracting numeric causal targets, cohort IDs, and edit contracts.\n"
        "- corrected_direct286 remains installed; factual dNLL +0.00045 on the latest cohort.\n"
        "- No weights changed.\n"
    )
```