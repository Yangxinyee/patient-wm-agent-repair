The diagnostic stopped because the CPU weight accessor is a function, not a dictionary. No weights changed. I’ll correct that call and finish checking whether dose contrasts can be isolated without disturbing factual predictions.

```python
import inspect, numpy as np, torch

def np191(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("WEIGHT ACCESSOR:", inspect.signature(w63),
      w63.__code__.co_names)
print("CPU CACHE VS SAVED EDITS:")
for name in edited():
    wc191 = np191(w63(name))
    print(name,
          "cache/edit RMS", float(np.sqrt(np.mean(
              (wc191 - np191(load(name)))**2))),
          "cache/original RMS", float(np.sqrt(np.mean(
              (wc191 - np191(load_original(name)))**2))))

print("\nCOUNTERFACTUAL INPUTS:")
for name in ["cf_features81", "cf_inputs96"]:
    obj = globals().get(name, {})
    print(name)
    for key, val in obj.items():
        if isinstance(val, dict):
            print(key, {k: tuple(v.shape) if hasattr(v, "shape")
                        else type(v).__name__ for k, v in val.items()})
        else:
            print(key, type(val).__name__,
                  tuple(val.shape) if hasattr(val, "shape") else "")

print("\nRELEVANT PIPELINE FUNCTIONS:")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and re.search(
        r"(63|73|74|81|82|83|96|173|176|177|178)$", name
    ):
        print(name, inspect.signature(obj))
        if name == "head_inputs96":
            print("names:", obj.__code__.co_names)
            print("constants:", obj.__code__.co_consts)

print("\nFACTUAL INPUT CHECK:")
x191 = np191(I180)
h191 = np191(H179)
print("inputs", x191.shape, "hidden", h191.shape,
      "events", np.unique(events179, return_counts=True),
      "fit rows", int(fitmask179.sum()))
hcheck191 = torch.nn.functional.gelu(
    torch.from_numpy(x191) @ w63("attr_heads.value_bin.0.weight").T
    + w63("attr_heads.value_bin.0.bias"))
print("hidden reconstruction RMS:",
      float(np.sqrt(np.mean((np191(hcheck191) - h191)**2))))
wout191 = np191(load_original("attr_heads.value_bin.2.weight"))
importance191 = np.std(h191.reshape(-1, 192), axis=0) * np.linalg.norm(
    wout191 - wout191.mean(axis=0, keepdims=True), axis=0)
quiet191 = np.argsort(importance191)[:16]
print("lowest-importance units:", quiet191.tolist())
print("importance:", np.round(importance191[quiet191], 5).tolist())

print("\nMETADATA OBJECTS:")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame) and re.search(
        r"(meta|read|obs|sample|panel|row)", name, re.I
    ):
        print(name, obj.shape, obj.columns.tolist()[:18])
```