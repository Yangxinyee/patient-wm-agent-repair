This repeats run 111’s output, so there is no new sensitivity result or weight change. I’ll retrieve the correction’s construction and the local forward helpers next. The hypothesis is that a better output direction—not merely a larger correction—can strengthen the MAP response while limiting factual disturbance.

```python
import dis, inspect, re
from pathlib import Path
import numpy as np
import torch

text112 = Path("/work/notes.md").read_text()
for section in re.split(r"(?=^## )", text112, flags=re.M):
    heading = section.splitlines()[0] if section.splitlines() else ""
    m = re.search(r"Run (\d+)", heading)
    if m and int(m.group(1)) in [46, 47, 81, 82, 103]:
        print(section[:900])

print("\n=== HELPER STRUCTURE ===")
for name in ["core_lin", "core_update_local", "head_calc", "capturing_core_lin46"]:
    fn = globals().get(name)
    if inspect.isfunction(fn):
        print(name)
        print("locals:", fn.__code__.co_varnames)
        print("dependencies:", fn.__code__.co_names)
        print("constants:", repr(fn.__code__.co_consts)[:1000])

print("\n=== RELEVANT OBJECTS ===")
for name, value in list(globals().items()):
    wanted = (
        name.startswith(("core_", "head_", "recon_", "flow_"))
        or re.search(r"(81|82|94|95|102|103)$", name)
        or name in ["W", "weights", "dt0", "state_x", "state_df",
                    "trial_direction47", "E47", "E0"]
    )
    if not wanted or callable(value):
        continue
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(name, tuple(value.shape))
    elif isinstance(value, dict):
        print(name, "dict", len(value), "keys", list(value)[:8])
    elif hasattr(value, "shape"):
        print(name, type(value).__name__, value.shape)
    elif isinstance(value, (float, int, str, tuple, list)):
        print(name, repr(value)[:130])

print("\n=== CORE UPDATE AFTER b_in ===")
instructions112 = list(dis.get_instructions(core_update_local))
start112 = next((i for i, ins in enumerate(instructions112)
                 if ins.opname == "LOAD_CONST" and ins.argval == "core.b_in"), None)
if start112 is not None:
    for ins in instructions112[max(0, start112-8):]:
        print(ins.opname, ins.argrepr)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 112: recover direction construction\n"
            "Run111 output was repeated; it supplied no new experiment. "
            "Inspecting detector/output-direction construction and forward-only helpers "
            "for a MAP-selective correction. No weights changed. "
            "Python runs: 112; forwards: 15.\n")
```