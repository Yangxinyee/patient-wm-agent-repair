The retained analyses support an immediate insulin effect near −2 per rate unit and a norepinephrine effect near +3, rather than the much weaker original rollouts. These are record-based estimates, not known truth. I’ll inspect the saved state and readout calculations next, so a head correction can target dose contrasts without replacing the model’s factual predictions.

```python
import inspect

print("=== HELPER GLOBAL DEPENDENCIES ===")
for name in ["replay_inputs161", "call_core161", "core_step136",
             "readout153", "readout_capture156", "capture159",
             "encode_actions168", "features169"]:
    obj = globals()[name]
    print(name, obj.__code__.co_names)

print("\n=== SOURCE STRINGS ===")
source_matches178 = []
for name, obj in list(globals().items()):
    if isinstance(obj, str) and name not in {"name"}:
        if any(s in obj for s in ["def readout153", "def replay_inputs161", "def core_step136"]):
            source_matches178.append((name, obj))
        elif len(obj) > 150 and any(s in name.lower() for s in ["src", "source", "code"]):
            print(name, len(obj), obj[:90].replace("\n", " "))
for name, text in source_matches178:
    print("\nSOURCE", name)
    print(text[:4200])

print("\n=== REFERENCED CACHE OBJECTS ===")
deps178 = set()
for name in ["replay_inputs161", "call_core161", "encode_actions168"]:
    deps178.update(globals()[name].__code__.co_names)
for name in sorted(deps178):
    obj = globals().get(name)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape), type(obj).__name__)
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj.keys())[:5])
    elif hasattr(obj, "columns"):
        print(name, obj.shape, list(obj.columns))
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))

print("\n=== RETAINED READOUT CAPTURES ===")
for event, cap in captured174.items():
    print(event, type(cap).__name__)
    if isinstance(cap, dict):
        print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
               for k, v in cap.items()])

print("\n=== ACTION / HEAD WEIGHT SCALE ===")
for name in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.2.weight",
             "core.c_out.weight", "last_numeric.weight"]:
    x = load_original(name)
    print(name, tuple(x.shape), "RMS", float(torch.as_tensor(x).square().mean().sqrt()))

print("\n=== NOTEBOOK: LAG IDENTIFICATION ===")
text178 = Path("/work/notes.md").read_text()
lines178 = text178.splitlines()
hits178 = [i for i, line in enumerate(lines178)
           if any(s in line.lower() for s in ["controlled insulin", "controlled lag", "lag estimate", "convolution"])]
for i in hits178[:5]:
    print("\n".join(lines178[max(0, i-2):i+5]))
```
