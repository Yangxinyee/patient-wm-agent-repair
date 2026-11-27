The slot-mean fits still show weak, asymmetric dose responses; they do not yet justify adding a MAP correction. The glucose-only edit remains our validated fallback. I’ll recover the exact replay interfaces and record-based effect estimates before changing it. A new route must improve dose responses while keeping factual NLL within the 0.005-nat limit.

```python
import pathlib, inspect, numpy as np, torch

def rr_recover():
    env = list(globals().items())
    print("=== ARCHITECTURE ===")
    print(pathlib.Path("/model/ARCHITECTURE.md").read_text())
    notes = pathlib.Path("/work/notes.md").read_text()
    print("\n=== EARLY IDENTIFICATION NOTES ===")
    print(notes[:9500])

    print("\n=== LOCAL FUNCTION INTERFACES ===")
    for name, obj in env:
        if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
            try:
                sig = inspect.signature(obj)
                params = [(p.name, str(p.kind)) for p in sig.parameters.values()]
                print(name, params)
            except Exception:
                print(name, "(signature unavailable)")

    print("\n=== RECENT DATA OBJECTS ===")
    found = []
    for name, obj in env:
        if name.startswith("_") or name.startswith("rr"):
            continue
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            found.append((name, type(obj).__name__, tuple(obj.shape)))
        elif isinstance(obj, dict):
            found.append((name, "dict", list(obj.keys())[:16]))
        elif hasattr(obj, "columns") and hasattr(obj, "shape"):
            found.append((name, "DataFrame", tuple(obj.shape), list(obj.columns)[:16]))
    for item in found[-100:]:
        print(item)

    print("\n=== CURRENT EDIT ===", edited())
    w = load("core.b_in.weight")
    a = w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
    np.savez("/work/recurrent_glucose_valid.npz", b_in=a)
    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 258: recover exact interfaces before further edits\n"
                "- Slot-mean feature fits alone do not resolve dose-response asymmetry.\n"
                "- Preserved the factual-validated glucose-only edit in recurrent_glucose_valid.npz.\n"
                "- Recovering architecture, identification evidence and retained helper interfaces.\n"
                "- Submitted weights unchanged. Python runs: 258; forward passes: 21.\n")

rr_recover()
```