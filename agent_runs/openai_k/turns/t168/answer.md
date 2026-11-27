The architecture confirms that dose effects must pass through the recurrent state, where their persistence is determined. I’ll restrict the next inspection to the interfaces needed for finite-difference calibration, avoiding another oversized dump. A successful calibration should reproduce the record-derived one-hour effects before we test longer rollouts.

```python
import inspect, dis, io, re, numpy as np, torch

def compact153(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        return "dict{" + ", ".join(
            str(k) + ":" + compact153(v)
            for k, v in list(x.items())[:10]
        ) + "}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    if hasattr(x, "coef_"):
        return f"{type(x).__name__}(coef={np.shape(x.coef_)})"
    if callable(x):
        return "callable"
    return repr(x)[:90]

print("=== ROUTING OBJECTS ===")
names153 = [
    n for n in sorted(list(globals()))
    if not n.startswith("_") and re.search(
        r"(145|151|Bcomp89|Binnov89|D89|V88|token_mu88|post66_base|impulse|deconv)", n, re.I
    )
]
for n153 in names153:
    if n153 in ("notes151", "chosen151", "choices151"):
        continue
    print(n153, compact153(globals()[n153]))

print("\n=== CALLABLE INTERFACES ===")
fnames153 = [
    n for n in sorted(list(globals()))
    if callable(globals()[n]) and hasattr(globals()[n], "__code__")
    and (
        re.search(r"(feature|readout|view|repaired|r59_core|145|counterfactual|physical)", n, re.I)
        or n in ("audit_core_update", "build_candidate89")
    )
]
for n153 in fnames153:
    f153 = globals()[n153]
    print(n153, str(inspect.signature(f153)),
          "locals", f153.__code__.co_varnames[:12],
          "globals", f153.__code__.co_names[:30])

print("\n=== LOCAL UPDATE SOURCE ===")
for n153 in ("repaired_core_local", "r59_core"):
    if n153 not in globals():
        continue
    try:
        src153 = inspect.getsource(globals()[n153])
    except Exception:
        buf153 = io.StringIO()
        dis.dis(globals()[n153], file=buf153)
        src153 = buf153.getvalue()
    print(n153, src153[:2400])

print("\n=== RETAINED READOUT DETAILS ===")
if "models145" in globals():
    for k153, m153 in models145.items():
        print(k153, compact153(m153))
if "diagnostics145" in globals():
    print(str(diagnostics145)[:1000])

print("\n=== IDENTIFICATION NOTEBOOK BEGINNING ===")
txt153 = open("/work/notes.md").read()
print(txt153[:3500])

with open("/work/notes.md", "a") as f153:
    f153.write(
        "\n## Run 153: compact finite-difference calibration interfaces\n"
        "- Architecture routes posterior dose innovations through z_in and the recurrent state.\n"
        "- Restricting inspection to exact local update, validated paired readouts, and retained identification estimates.\n"
        "- No weights changed. Python runs: 153; forward passes: 10.\n"
    )
```