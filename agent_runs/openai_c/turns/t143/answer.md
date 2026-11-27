The bin edit did not strengthen the responses and substantially weakened the insulin-halving response. I’ll remove it and review the saved state directions for a narrower recurrent edit. The earlier broad recurrent gain improved insulin sensitivity but failed factual fit; a selective version would need to retain that benefit without the penalty.

```python
from pathlib import Path
import numpy as np
import inspect

for _ in range(30):
    undo()
print("RESTORED OPERATIONS", ops())

notes = Path("/work/notes.md").read_text()
print("\nNOTEBOOK SECTION HEADINGS")
for line in notes.splitlines():
    if line.startswith("## "):
        print(line)

print("\nSAVED FUNCTIONS")
for key, value in sorted(list(globals().items())):
    if inspect.isfunction(value) and getattr(value, "__module__", "") == "__main__":
        try:
            print(key, inspect.signature(value))
        except Exception:
            print(key)

print("\nSTATE/DIRECTION ARRAYS")
for key, value in sorted(list(globals().items())):
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        if any(s in (32, 192, 768, 992) for s in value.shape):
            print(key, value.shape)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 101 / forward 39: bin amplification rejected\n"
            "- Bin-contrast gain 3: insulin -2.128/+0.339; NE +0.059/-0.357.\n"
            "- No useful improvement; removed all 30 bin operations.\n"
            "- Numeric gain 4 alone restored.\n"
            "- Reviewing saved state directions for selective recurrent amplification.\n"
            "- Forward passes used: 39; 11 remain.\n")
```