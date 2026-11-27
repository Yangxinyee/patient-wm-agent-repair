The retained NE detector uses only 192 prelude features and still reacts to insulin edits. I’ll test the prelude feed-forward layer’s 768 nonlinear features instead. They may support a more selective detector, though extra features alone do not guarantee success. Lower held-out factual leakage and balanced NE contrasts would justify a reversible weight trial.

```python
import dis, inspect, numpy as np, torch
from pathlib import Path

print("PRELUDE HELPERS")
for name in ["prelude46", "core_layer", "core_ffn", "core_norm", "coda46"]:
    fn = globals()[name]
    print("\n", name, inspect.signature(fn))
    print("locals", fn.__code__.co_varnames, "constants", fn.__code__.co_consts)
    try:
        print(inspect.getsource(fn))
    except (OSError, TypeError):
        instructions = list(dis.get_instructions(fn))
        compact = [f"{i.opname}:{i.argrepr}" for i in instructions
                   if i.opname not in {"RESUME", "CACHE", "NOP"}]
        for start in range(0, len(compact), 12):
            print(" | ".join(compact[start:start+12]))

print("\nMATCHING INPUT ARRAYS")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and obj.ndim:
        if obj.shape[0] in [4356, 1497] and (
            name.startswith(("core", "recon", "cf", "_act", "_u", "_U")) or
            "46" in name or "47" in name
        ):
            print(name, tuple(obj.shape))
    elif isinstance(obj, dict) and any(s in name for s in [
        "slot_targets", "slot_deltas", "configs", "memories"
    ]):
        print(name, "keys", list(obj)[:12])

print("\nSPLIT MASKS / TARGET SHAPES")
for name in ["train_cf46", "test_cf46", "train_slots81",
             "D81", "Y81", "slot_targets47", "slot_deltas47"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {str(k): getattr(v, "shape", None) for k, v in obj.items()})
    elif obj is not None:
        print(name, getattr(obj, "shape", None), str(obj)[:140])

print("\nBASELINE VALIDATION NOTE")
for section in Path("/work/notes.md").read_text().split("\n## "):
    if "Run 68:" in section.splitlines()[0]:
        print(("## " + section)[:1600])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 92: nonlinear prelude feature route\n"
        "Considering a closed-form detector on the prelude FFN's 768 hidden "
        "features rather than the 192-dimensional normalized prelude. "
        "Need held-out factual-null, admission-null, drug-specific and balanced "
        "contrast checks before a reversible trial. No weights changed. "
        "Python runs: 92; forwards: 12.\n"
    )
```