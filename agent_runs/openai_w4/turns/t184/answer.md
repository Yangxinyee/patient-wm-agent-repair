The architecture confirms that dose edits must work through recurrent state, while generated readings also feed back through the last-value register. The early record analyses support an insulin-lowering effect, but do not by themselves establish its 12-hour magnitude. I’ll recover the retained candidates and replay interfaces with compact output so the interrupted persistence test can actually run.

```python
import inspect

print("CURRENT EDITS", edited())
print("\nREPLAY / ACTION / IMPULSE FUNCTIONS")
for nm163 in sorted(list(globals())):
    ob163 = globals()[nm163]
    if inspect.isfunction(ob163) and any(s in nm163.lower() for s in
        ["replay", "impulse", "action", "transition", "reading", "context", "nll"]):
        try:
            print(nm163, inspect.signature(ob163))
        except Exception:
            print(nm163)
        if nm163 in ["replay_factual125", "transition", "transition125"]:
            print("  dependencies:", ob163.__code__.co_names)

print("\nRECENT VARIABLE NAMES")
print([nm for nm in sorted(list(globals()))
       if nm.endswith(("159", "160", "161"))])

print("\nIMPORTANT OBJECTS")
for nm163 in ["repair_bank", "weights_channel125", "weights_rejected125",
              "impulse_results159", "scale_results128", "lp_new136"]:
    if nm163 not in globals():
        continue
    ob163 = globals()[nm163]
    if isinstance(ob163, dict):
        print(nm163, {
            str(k): (tuple(v.shape) if hasattr(v, "shape")
                     else f"{type(v).__name__}[{len(v)}]" if hasattr(v, "__len__")
                     else type(v).__name__)
            for k, v in list(ob163.items())[:24]
        })
    else:
        print(nm163, type(ob163).__name__,
              tuple(ob163.shape) if hasattr(ob163, "shape") else "")

print("\nREPLAY SOURCE IF AVAILABLE")
if "replay_factual125" in globals():
    try:
        print(inspect.getsource(replay_factual125)[:3600])
    except Exception:
        print("arguments/local names:", replay_factual125.__code__.co_varnames)
        print("constants:", [x for x in replay_factual125.__code__.co_consts
                             if isinstance(x, (str, int, float, tuple))][:30])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 163\nRecovered architecture in run 162, but its output "
            "truncation hid the retained interfaces. Using compact inspection now. "
            "Record threshold and dose-innovation evidence supports insulin lowering; "
            "long-horizon magnitude still requires dynamic identification. "
            "No saved weights changed.\n")
```