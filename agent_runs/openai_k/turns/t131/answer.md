The recovered notes show that history-adjusted estimates differ substantially from simple associations, especially for norepinephrine. They support insulin lowering glucose and norepinephrine raising MAP, but the longer-term response still needs to guide the edit. I’ll recover those dynamic estimates and the retained repair functions; agreement across time and sensors would strengthen the targets, while inconsistent dynamics would argue against calibrating to a single slope.

```python
import re, inspect
notes122 = Path("/work/notes.md").read_text()
start122 = notes122.find("## Run 6:")
stop122 = notes122.find("## Run 13:")
print("=== EARLY DYNAMICS AND REPAIR EVIDENCE ===")
print(notes122[start122:stop122][:4800])
print("\n=== RETAINED FUNCTION NAMES ===")
fn122 = []
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__" and not k.startswith("_"):
        try:
            fn122.append(k + str(inspect.signature(v)))
        except Exception:
            fn122.append(k)
print("\n".join(fn122)[:2300])
print("\n=== CANDIDATE / TARGET OBJECTS ===")
for k, v in list(globals().items()):
    if re.search(r"candidate|target|impulse|effect|baseline|response|sensitiv", k, re.I):
        if isinstance(v, dict):
            print(k, "keys:", list(v.keys())[:12])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape), str(v)[:180] if v.ndim <= 1 else "")
        elif isinstance(v, (list, tuple)):
            print(k, "length:", len(v), str(v)[:180])
print("\n=== RELEVANT WEIGHTS ===")
for n in NAMES:
    if (n.startswith(("embed.numeric", "encode_action", "core.z_", "core.flow"))
        or "last_numeric" in n or "attr_heads.value" in n):
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 122: dynamic-target recovery\n"
            "- Cross-fitted immediate estimates support insulin glucose effects near -2 and NE MAP effects near +3 per rate unit.\n"
            "- Recovering persistence and retained repair interfaces; no edits submitted.\n"
            "- Python runs: 122; forward passes: 4.\n")
```
