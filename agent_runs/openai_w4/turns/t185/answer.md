The complete replay and original-state fixture are available, which lets us test recurrence without guessing from isolated transitions. The submitted edit is still unchanged. I’ll identify the action tensor and summarize the retained connection tests, then repeat the interrupted impulse test using tensor-safe copies. The key test is whether a factually stable connection preserves a dose response beyond the next reading.

```python
import dis

print("REPLAY IMPLEMENTATION")
dis.dis(replay_factual125)

print("\nTRANSITION DEPENDENCIES")
print("transition73:", transition73.__code__.co_names)
print("transition73 locals:", transition73.__code__.co_varnames)

print("\nRP INTERFACES")
for nm164 in sorted(list(globals())):
    if not nm164.startswith("rp_"):
        continue
    ob164 = globals()[nm164]
    if inspect.isfunction(ob164):
        print(nm164, inspect.signature(ob164), "deps:", ob164.__code__.co_names)
    elif isinstance(ob164, dict):
        print(nm164, {str(k): tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                      for k, v in list(ob164.items())[:15]})
    elif hasattr(ob164, "shape"):
        print(nm164, type(ob164).__name__, tuple(ob164.shape))

print("\nRECENT DIAGNOSTICS")
for nm164 in ["direction161", "delta160", "saved_u160", "new_u160",
              "chosen160", "scale_tests161", "connection_tests160"]:
    if nm164 not in globals():
        continue
    ob164 = globals()[nm164]
    if torch.is_tensor(ob164) or isinstance(ob164, np.ndarray):
        ar164 = ob164.detach().cpu().numpy() if torch.is_tensor(ob164) else ob164
        print(nm164, type(ob164).__name__, ar164.shape,
              "norm", float(np.linalg.norm(ar164)), "finite", bool(np.isfinite(ar164).all()))
    elif isinstance(ob164, dict):
        print(nm164, str(ob164)[:1800])
    elif isinstance(ob164, (list, tuple)):
        print(nm164, str(ob164)[:1800])
    else:
        print(nm164, repr(ob164)[:600])

def independent_copy164(x):
    return x.detach().clone() if torch.is_tensor(x) else x.copy()

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 164\nInspecting exact action/replay dependencies and retained "
            "connection tests. Defined tensor-safe copying for the interrupted "
            "impulse experiment. No saved weights changed.\n")
```