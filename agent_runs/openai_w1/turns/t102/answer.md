The retained trace fits use very large coefficients, so their small reported errors are not enough to establish robustness—especially after correcting the input encodings. I won’t reuse them unchecked.

Next I’ll identify their construction and the latent-posterior tensor shapes. A useful edit should respond consistently to dose scaling without amplifying tiny reconstruction errors.

```python
import re
import inspect
import numpy as np
from pathlib import Path

r93_notes = Path("/work/notes.md").read_text()
r93_sections = re.split(r"(?m)(?=^## )", r93_notes)
print("=== DEVELOPMENT SUMMARY, RUNS 30–58 ===")
for r93_section in r93_sections:
    r93_header = r93_section.split("\n", 1)[0]
    r93_numbers = [int(x) for x in re.findall(r"\d+", r93_header)]
    if r93_numbers and 30 <= r93_numbers[0] <= 58:
        print(r93_section[:430].rstrip(), "\n")

print("=== POSTERIOR AND ACTION TENSOR SHAPES ===")
for r93_name in NAMES:
    if any(s in r93_name for s in ("z_post", "z_in", "encode_action.mlp2", "attr_heads.value_bin")):
        print(r93_name, SHAPES[r93_name])

print("\n=== HELPER DEPENDENCIES ===")
for r93_name in ["rr50_action_parts", "repair_action_encoding", "r89_encode_actions",
                 "r88_local_atoms", "r72_posterior"]:
    r93_fun = globals().get(r93_name)
    if inspect.isfunction(r93_fun):
        print(r93_name, "globals:", r93_fun.__code__.co_names,
              "locals:", r93_fun.__code__.co_varnames)
        try:
            r93_source = inspect.getsource(r93_fun)
            print(r93_source[:700])
        except Exception:
            pass

print("\n=== AVAILABLE SOURCE STRINGS ===")
for r93_name, r93_obj in list(globals().items()):
    if isinstance(r93_obj, str) and len(r93_obj) > 200 and (
        "def " in r93_obj or "rr51" in r93_obj or "rr50" in r93_obj
    ):
        print(r93_name, "characters", len(r93_obj), "start", repr(r93_obj[:85]))

with open("/work/notes.md", "a") as r93_f:
    r93_f.write(
        "\n## Run 93\n"
        "Retained trace decoders have very large coefficient norms; require recalibration and extrapolation checks "
        "with corrected encodings before reuse. Inspecting posterior/action tensor layout and candidate construction. "
        "No tensors edited. Counts: 93 Python runs, 6 forward passes.\n"
    )
```
