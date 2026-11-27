The helpers are available, and the earlier CPU screen found a gate-27, norepinephrine-gain-1.4 variant within the factual budget. I’ll save that variant for testing. The hypothesis is that it will moderate insulin’s overshoot and strengthen norepinephrine’s response; only the paired rollout and factual checks can confirm that.

```python
import torch, inspect, re
from pathlib import Path

backup294 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
chosen294 = (3.0, 0.0, 27.0, 1.4)
candidate294 = asymmetric_ins279(
    chosen294[0], chosen294[1], chosen294[2], ne_gain=chosen294[3]
)
assert all(n in candidate294 for n in NAMES)
oldnames294 = set(edited())
changed294 = []
for n in NAMES:
    value294 = torch.as_tensor(candidate294[n], dtype=torch.float32).clone()
    original294 = torch.as_tensor(load_original(n), dtype=torch.float32)
    assert tuple(value294.shape) == tuple(SHAPES[n])
    assert torch.isfinite(value294).all()
    if not torch.equal(value294, original294):
        save_edit(n, value294)
        changed294.append(n)
    elif n in oldnames294:
        drop_edit(n)
print("SAVED_CANDIDATE", chosen294)
print("EDITED_TENSORS", changed294)
print("PRIOR_CPU_SCREEN_DNLL", 0.004330039024353027)

print("\nCPU_AND_CONSTRUCTION_HELPERS")
for n, o in list(globals().items()):
    if inspect.isfunction(o) and (
        re.search(r"factual|score|teacher|evaluate|fixture|sequence|loss|symmetric", n, re.I)
        or n.startswith("rp_")
    ):
        print(n, inspect.signature(o))

print("\nFIXTURE_VARIABLES")
fixture_lines294 = []
for n, o in list(globals().items()):
    if re.search(r"(194|258|265|272)$", n) and not inspect.isfunction(o):
        if hasattr(o, "shape"):
            description294 = str(o.shape)
        elif isinstance(o, dict):
            description294 = f"dict({len(o)}), keys={list(o)[:6]}"
        elif isinstance(o, (list, tuple)):
            description294 = f"{type(o).__name__}({len(o)})"
        else:
            description294 = str(o)[:100]
        fixture_lines294.append(f"{n}: {description294}")
print("\n".join(fixture_lines294)[:2400])

lines294 = Path("/work/notes.md").read_text().splitlines()
evidence_lines294 = [
    (i, s) for i, s in enumerate(lines294)
    if re.search(r"DML|confound|negative.control|longitudinal|identifi|kernel.*target", s, re.I)
    and not re.search(r"recover|pending|unchanged|transmission|hypothes", s, re.I)
]
print("\nRECORD_IDENTIFICATION_EXCERPTS")
for i, s in evidence_lines294[:5]:
    print("\n".join(lines294[max(0, i-1):i+3])[:350])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 294\nSaved asymmetric insulin 3/0, gate 27, earlier-NE gain 1.4. "
            "Prior same-fixture CPU dNLL +0.00433004. Run-293 submission backed up "
            "as backup294. Hypothesis: reduce insulin overshoot while strengthening "
            "NE; GPU rollout and factual checks pending. No new causal estimates.\n")
```