The captured inputs let us test a new head without rerunning the whole model. My hypothesis is that dose information is present but poorly separated from patient-condition features. I’ll check that signal and identify replaceable hidden units; accurate dose contrasts with small factual residuals would support a selective repair.

```python
import inspect, numpy as np, torch

print("CPU CACHE VS SAVED EDITS:")
for name in edited():
    if name in w63:
        print(name,
              "cache/edit RMS", float(torch.sqrt(torch.mean(
                  (w63[name].float() - load(name).float())**2))),
              "cache/original RMS", float(torch.sqrt(torch.mean(
                  (w63[name].float() - load_original(name).float())**2))))

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
        if name in ["head_inputs96"]:
            print("names:", obj.__code__.co_names)
            print("constants:", obj.__code__.co_consts)

print("\nFACTUAL INPUT CHECK:")
def np190(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
x190 = np190(I180)
h190 = np190(H179)
print("inputs", x190.shape, "hidden", h190.shape,
      "events", np.unique(events179, return_counts=True),
      "fit rows", int(fitmask179.sum()))
hcheck190 = torch.nn.functional.gelu(
    torch.from_numpy(x190) @ w63["attr_heads.value_bin.0.weight"].T
    + w63["attr_heads.value_bin.0.bias"])
print("hidden reconstruction RMS:",
      float(np.sqrt(np.mean((np190(hcheck190) - h190)**2))))
wout190 = np190(load_original("attr_heads.value_bin.2.weight"))
importance190 = np.std(h190.reshape(-1, 192), axis=0) * np.linalg.norm(
    wout190 - wout190.mean(axis=0, keepdims=True), axis=0)
quiet190 = np.argsort(importance190)[:16]
print("lowest-importance units:", quiet190.tolist())
print("importance:", np.round(importance190[quiet190], 5).tolist())

print("\nMETADATA OBJECTS:")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame) and re.search(
        r"(meta|read|obs|sample|panel|row)", name, re.I
    ):
        print(name, obj.shape, obj.columns.tolist()[:18])

print("\nCURRENT SNAPSHOT RETAINED:", list(snapshot188))
```