The recovered notes support an insulin effect near −1.94 per unit and a norepinephrine effect near +3.05 per rate unit; the simpler state-space estimates failed the causal checks. The architecture also confirms that doses enter upstream of the recurrent state. I’ll retrieve the validated computation interfaces to test a dose-residual correction, rather than amplify the unsuccessful head fit.

```python
import inspect, re
from pathlib import Path

snap99 = dict(globals())
print("=== CUSTOM FUNCTIONS ===")
lines99 = []
for n99, o99 in sorted(snap99.items()):
    if inspect.isfunction(o99) and getattr(o99, "__module__", "") == "__main__":
        try:
            lines99.append(f"{n99}{inspect.signature(o99)}")
        except Exception:
            lines99.append(n99)
print("\n".join(lines99)[:4400])

print("\n=== ACTION, LATENT, AND VALUE WEIGHTS ===")
for n99 in NAMES:
    if re.search(r"encode_action|z_post|z_prior|z_in|z_head|value_bin|embed\.numeric|last_numeric|type_emb|query_norm", n99):
        print(n99, SHAPES[n99])

print("\n=== RELEVANT RETAINED VARIABLES ===")
lines99 = []
for n99, o99 in sorted(snap99.items()):
    if re.search(r"(?:64|75|76|82|83|88|94|96|97)$|causal|kernel|effects14|rich_fits", n99):
        s99 = getattr(o99, "shape", None)
        if s99 is not None:
            lines99.append(f"{n99}: {type(o99).__name__} {tuple(s99)}")
        elif isinstance(o99, dict):
            lines99.append(f"{n99}: dict {list(o99)[:10]}")
        elif isinstance(o99, (list, tuple)):
            lines99.append(f"{n99}: {type(o99).__name__} length {len(o99)}")
print("\n".join(lines99)[:2900])

with Path("/work/notes.md").open("a") as f99:
    f99.write(
        "\n## Run 99 / compact interface recovery\n"
        "Architecture confirms doses enter through the action encoder and posterior/prelude. "
        "Recovered notes support corrected DML effects I≈-1.94 and NE≈+3.05; rejected SSMs "
        "must not determine targets. Retrieving validated functions and edit locations. No edits.\n"
    )
```
