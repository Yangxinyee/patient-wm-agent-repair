The history-controlled and dynamic estimates consistently support a positive norepinephrine effect, though the rollout target remains assumption-dependent. I’ll test a joint gain adjustment: slightly reduce the insulin correction toward its estimated target and use the freed factual-loss margin for MAP. Only candidates passing stochastic held-out checks will proceed to GPU validation.

```python
import dis
import numpy as np
import torch

print("MAP BUILDER")
dis.dis(head177_make)

print("\nMONTE CARLO INPUT DICTIONARIES")
mc_inventory183 = {}
for name, obj in sorted(list(globals().items())):
    if not isinstance(obj, dict) or "inputs" not in obj or "logits" not in obj:
        continue
    x183 = obj["inputs"]
    if hasattr(x183, "shape"):
        print(name, "inputs", tuple(x183.shape),
              "logits", tuple(obj["logits"].shape),
              "keys", list(obj.keys()))
        if len(x183.shape) == 3:
            mc_inventory183[name] = obj

print("\nFIT OBJECT")
fit183 = map164_fits[(0.0, 1.0, 0.001)]
print(type(fit183), tuple(fit183.shape) if hasattr(fit183, "shape") else fit183.keys())
print("MAP FREE UNITS:", head177_free)

print("\nINSULIN UNITS")
for unit183 in [170, 89]:
    original_col183 = np.asarray(best_insulin_v1["attr_heads.value_bin.2.weight"])[:, unit183]
    print(unit183, "saved output-column norm", np.linalg.norm(original_col183),
          "current first-layer difference",
          np.max(np.abs(
              np.asarray(load("attr_heads.value_bin.0.weight"))[unit183] -
              np.asarray(best_insulin_v1["attr_heads.value_bin.0.weight"])[unit183]
          )))

print("\nCURRENT TRAIN EVALUATION")
eval183 = head177_eval(map164_fact, head177_weights)
for key183, value183 in eval183.items():
    a183 = np.asarray(value183)
    print(key183, a183.shape,
          "mean", float(a183.mean()),
          "sd", float(a183.std()))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 183: joint-gain preparation\n"
        "- NE history-controlled, dynamic-panel and proxy estimates support a positive effect; exact persistence remains uncertain.\n"
        "- Proposed next test: reduce insulin gain slightly and strengthen MAP within stochastic held-out factual constraints.\n"
        "- No new edits installed; combined v1 retained.\n"
    )
```