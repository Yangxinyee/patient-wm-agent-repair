The architecture confirms that dose changes can affect predictions through both recurrent state and generated readings fed back into the model. The notes also show a delayed insulin response in the records, so an immediate-effect calibration is insufficient. I’ll retrieve the fitted response kernels and dose-transmission diagnostics; poor transmission would support repairing the channel first, while good transmission would implicate rollout dynamics.

```python
def r215_brief(x, depth=0):
    if hasattr(x, "shape"):
        return {"shape": tuple(x.shape)}
    if isinstance(x, dict):
        if depth >= 2:
            return {"keys": list(x)[:16]}
        return {str(k): r215_brief(v, depth + 1) for k, v in list(x.items())[:16]}
    if isinstance(x, (list, tuple)):
        if len(x) <= 12:
            return [r215_brief(v, depth + 1) for v in x]
        return {"length": len(x)}
    if isinstance(x, (int, float, str, bool, np.number)) or x is None:
        return x
    return type(x).__name__

print("=== COMPACT INTERFACES ===")
for r215_name, r215_obj in sorted(list(globals().items())):
    if r215_name.startswith(("r214_", "r215_")):
        continue
    if callable(r215_obj) and hasattr(r215_obj, "__code__") and re.search(
        r"dual|candidate|direction|factual|rollout|causal|target|dose_metrics|gated_ne", r215_name
    ):
        print(r215_name, str(inspect.signature(r215_obj))[:200])

print("\n=== EFFECT AND TRANSMISSION OBJECT NAMES ===")
r215_names = [
    n for n, o in list(globals().items())
    if not n.startswith(("r214_", "r215_")) and not callable(o)
    and re.search(r"target|causal|effect|kernel|calibrations192|results194|result87|rate192|active192|gate_fits180", n, re.I)
]
print(r215_names)
for r215_name in r215_names:
    if re.search(r"target|kernel|calibrations192|results194|result87", r215_name, re.I):
        print(r215_name, repr(r215_brief(globals()[r215_name]))[:1200])

print("\n=== LONGITUDINAL EVIDENCE NOTES ===")
r215_selected = [
    s for s in r214_sections
    if re.search(r"two.pole|six.lag|13\.2008|3\.6087|long.tail|equal.stay|bootstrap", s, re.I)
]
print("\n".join(r215_selected)[:3500])

with open("/work/notes.md", "a") as r215_file:
    r215_file.write(
        "\n## Run 215\n"
        "Retrieved compact repair interfaces, record-effect/kernel objects and dose-transmission diagnostics "
        "after Run-214 output truncation. Architecture confirms recurrent and generated-reading feedback; "
        "delayed record-derived insulin effects require longitudinal rather than immediate calibration. "
        "No saved weights changed.\n"
    )
```