The architecture confirms that dose edits affect the recurrent state, not just the value register. The output was truncated before the retained functions appeared. I’ll retrieve those functions and the accepted effect estimates compactly so the next check uses the previously validated implementation.

```python
import inspect
from pathlib import Path

print("=== FUNCTIONS (compact) ===")
lines69 = []
for n69, o69 in list(globals().items()):
    if inspect.isfunction(o69) and getattr(o69, "__module__", "") == "__main__":
        try:
            s69 = str(inspect.signature(o69))
        except Exception:
            s69 = "?"
        lines69.append(f"{n69}{s69}")
print("\n".join(lines69)[:3000])

print("\n=== RECENT OBJECTS ===")
lines69 = []
for n69, o69 in list(globals().items()):
    if n69.endswith(("65", "66", "67")) or n69 in (
        "kernels19", "r32_best", "e_valid60", "u_valid60",
        "state_train_ids36", "d22", "P", "A"
    ):
        if hasattr(o69, "shape"):
            desc69 = f"{type(o69).__name__} {tuple(o69.shape)}"
            if hasattr(o69, "columns"):
                desc69 += " columns=" + repr(list(o69.columns))
        elif isinstance(o69, dict):
            desc69 = "keys=" + repr(list(o69)[:30])
        elif isinstance(o69, (list, tuple)):
            desc69 = f"{type(o69).__name__} length={len(o69)}"
        elif inspect.isfunction(o69):
            continue
        else:
            desc69 = repr(o69)[:130]
        lines69.append(f"{n69}: {desc69}")
print("\n".join(lines69)[:2200])

print("\n=== ACCEPTED CAUSAL ANALYSES ===")
text69 = Path("/work/notes.md").read_text()
start69 = text69.find("## Run 12")
end69 = text69.find("## Run 21", start69)
print(text69[start69:end69 if end69 >= 0 else start69 + 3000][:3000])

print("\n=== KERNEL ESTIMATES ===")
if "kernels19" in globals():
    print(repr(kernels19)[:1200])

with Path("/work/notes.md").open("a") as f69:
    f69.write(
        "\n## Run 69\n"
        "Architecture recovered; run 68 output truncated before function inventory. "
        "Retrieving compact retained interfaces and accepted causal kernels. No edits.\n"
    )
```