The records support provisional targets near −10.5 for doubled insulin and +3.8 for doubled norepinephrine, but the current rollout reaches only +0.85 for norepinephrine. The latest frontier does not clearly improve the arterial readout. I’ll check how its scores are defined and verify that the installer reproduces the active weights before changing anything.

```python
import dis

print("=== HELPER DEPENDENCIES ===")
for _n in ["head177_make", "head177_eval", "screen_map218",
           "factual_curvature218", "scalar_mean192"]:
    _fn = globals()[_n]
    print(_n, "locals:", _fn.__code__.co_varnames)
    print("  names:", _fn.__code__.co_names)

print("\n=== INSTALLER BYTECODE ===")
print(dis.Bytecode(head177_make).dis()[:6500])

print("\n=== ACTIVE-WEIGHT REPRODUCTION ===")
_r221_rebuilt = head177_make(map192_candidates[(10.0, 6.0)])
for _n in head177_names:
    _a = np.asarray(load(_n))
    _b = np.asarray(_r221_rebuilt[_n])
    print(_n, "max absolute difference", float(np.max(np.abs(_a - _b))))
print("free neurons:", head177_free)
print("candidate coefficient:", type(moment218_candidates[
    ("rescaled", "double_focus", 0.001, 0.03, 0.5)]).__name__,
    getattr(moment218_candidates[
        ("rescaled", "double_focus", 0.001, 0.03, 0.5)], "shape", None))

print("\n=== CAUSAL EVIDENCE ===")
for _file in ["dynamic_iv.csv", "proxy_iv.csv", "sensor_iv.csv"]:
    _df = pd.read_csv("/work/" + _file)
    print(_file, _df.shape, list(_df.columns))
    print(_df.head(5).to_string(index=False))

_r221_sections = re.split(r"(?m)(?=^## )", open("/work/notes.md").read())
for _s in _r221_sections:
    if _s.startswith("## Causal identification checks"):
        print(_s[:1800])
```