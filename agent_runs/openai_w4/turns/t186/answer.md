The replay captures the treatment channel before the recurrent connection, and the action input is a tensor—explaining the copy error. I’ll now distinguish the tested response directions and recover the value-head helper. This will let the next experiment measure actual glucose predictions across lags, rather than treating channel size or state movement as evidence of persistence.

```python
print("ACTION AND VALUE HELPERS")
for nm165 in sorted(list(globals())):
    ob165 = globals()[nm165]
    if inspect.isfunction(ob165) and any(s in nm165.lower() for s in
        ["atom", "bin", "prob", "predict", "value", "head", "numeric"]):
        print(nm165, inspect.signature(ob165),
              "dependencies:", ob165.__code__.co_names)

print("\nIMPULSE FIXTURES")
for nm165 in ["E65", "U65", "ix160", "rates160", "doses160", "direction160",
              "ridge160", "decoder160", "chosen159", "atoms159",
              "new_u159", "pi160", "qi160", "lags160", "lag_masks160"]:
    if nm165 not in globals():
        continue
    ob165 = globals()[nm165]
    if torch.is_tensor(ob165) or isinstance(ob165, np.ndarray):
        print(nm165, type(ob165).__name__, tuple(ob165.shape))
        if getattr(ob165, "ndim", 2) == 1 and len(ob165) <= 35:
            print(" ", ob165.detach().cpu().numpy() if torch.is_tensor(ob165) else ob165)
    elif isinstance(ob165, dict):
        print(nm165, {str(k): tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                      for k, v in ob165.items()})
    else:
        print(nm165, type(ob165).__name__, str(ob165)[:500])

print("\nCONNECTION TESTS")
if "connection_tests160" in globals():
    print(str(connection_tests160)[:3500])

print("\nBASE IMPULSE RESULTS")
for drug165, result165 in impulse_results159.items():
    print(drug165, {
        str(k): (v.tolist() if isinstance(v, np.ndarray) and v.size <= 30
                 else v if np.isscalar(v)
                 else str(v)[:700])
        for k, v in result165.items()
    })

print("\nWEIGHT CANDIDATE RELATIONSHIPS")
for nm165 in ["channel_weights160", "original_weights160", "weights161"]:
    if nm165 in globals():
        w165 = globals()[nm165]["core.b_in.weight"]
        print(nm165, "B norm", float(torch.linalg.vector_norm(w165)),
              "distance from saved", float(torch.linalg.vector_norm(
                  w165 - torch.as_tensor(load("core.b_in.weight")))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 165\nRecovered replay's action input U65 and treatment-channel "
            "capture. Compact inspection distinguishes connection candidates and "
            "recovers the prediction head for lagged impulse testing. "
            "No saved weights changed.\n")
```