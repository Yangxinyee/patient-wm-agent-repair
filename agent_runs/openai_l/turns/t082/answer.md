The posterior matches, and the factual decoder means look consistent, but the recurrent state still does not match. I’ll test the memory-mask convention before using any counterfactual states. A return to the previously achieved reconstruction precision would confirm the fix.

```python
import numpy as np
import torch
import dis
from pathlib import Path

print("=== MEMORY-MASK CONVENTION TEST ===")
results75 = {}
for label75, candidate75 in [
    ("True means padding", mask74),
    ("True means valid", ~mask74),
]:
    oo75, hh75 = transition64(
        hf74, e_valid60[checkidx73], u72[checkidx73], post74,
        mem74, age74, candidate75
    )
    errors75 = (
        float((oo75-sx63[checkidx73, :192]).square().mean().sqrt()),
        float((hh75-sx63[checkidx73, 192:960].reshape(-1, 4, 192))
              .square().mean().sqrt()),
    )
    results75[label75] = errors75
    print(label75, "out/h RMS:", errors75)

print("\n=== ATTENTION MASK IMPLEMENTATION ===")
instructions75 = list(dis.get_instructions(attention63))
for position75, instruction75 in enumerate(instructions75):
    if instruction75.argval == "mask":
        print("\n".join(
            f"{i.offset}: {i.opname} {i.argrepr}"
            for i in instructions75[
                max(0, position75-3):min(len(instructions75), position75+22)
            ]
        ))

print("\n=== RETAINED MEMORY OBJECTS ===")
for n75, o75 in list(globals().items()):
    if n75.endswith(("63", "64")) and isinstance(o75, (np.ndarray, torch.Tensor)):
        print(n75, tuple(o75.shape), str(o75.dtype))
print("\n=== TRANSITION TAIL ===")
print("\n".join(
    f"{i.offset}: {i.opname} {i.argrepr}"
    for i in dis.get_instructions(transition64) if i.offset >= 528
)[:1600])

with Path("/work/notes.md").open("a") as f75:
    f75.write(
        "\n## Run 75 / recurrent memory-mask audit\n"
        "Posterior RMS=2.15e-5; factual decoder means track the observed means. "
        "Recurrent out RMS=.1306 and h RMS=12.29 are not acceptable. Testing "
        "mask polarity and inspecting retained validated memory inputs before "
        "using counterfactual state features. No edits.\n"
    )
```