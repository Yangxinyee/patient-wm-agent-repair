The stronger edit failed the main test: insulin doubling stayed near the original −1.85, despite acceptable factual fit. Halving improved only modestly, so edit strength does not translate reliably into rollout effect. I’ll restore the weaker checkpoint and inspect the hourly behavior and graft construction; saturation or feedback distortion would explain this failure.

```python
import inspect

graft_checkpoint_beta24 = {n: np.array(load(n), copy=True) for n in graft_names}
for n in graft_names:
    save_edit(n, graft_checkpoint_beta2[n])
graft_beta = -2.0

print("=== ROLLOUT CSV ===")
_r11 = pd.read_csv("/work/fwd_11/dose_edit_edited.csv")
print("columns:", list(_r11.columns))
print(_r11.head(8).to_string(index=False))
print("=== GRAFT / CAUSAL FUNCTION METADATA ===")
for _fnname in ["make_insulin_graft", "graft_probs", "head_average_probs3",
                "causal_shift_table", "insulin_rd", "make_ne_data", "rd_dynamic_residual"]:
    _fn = globals()[_fnname]
    print(_fnname, inspect.signature(_fn))
    print("variables:", _fn.__code__.co_varnames)
    print("globals:", _fn.__code__.co_names)
    print("constants:", repr(_fn.__code__.co_consts)[:600])
print("=== OTHER RETAINED NAMES ===")
print(" ".join(sorted(k for k in globals() if not k.startswith("_"))))
print("=== STORED CODE STRINGS ===")
for k, v in list(globals().items()):
    if not k.startswith("_") and isinstance(v, str) and len(v) > 300:
        print(k, "length", len(v), repr(v[:130]))
print("=== INSULIN RD NOTEBOOK SECTIONS ===")
_note = open("/work/notes.md").read()
for _needle in ["Regression discontinu", "Regression-discontinu", "Insulin RD", "insulin RD",
                "threshold", "RD-derived", "Dynamic", "dynamic RD"]:
    _pos = _note.find(_needle)
    if _pos >= 0:
        print(_note[max(0, _pos-120):_pos+1400])
with open("/work/notes.md", "a") as f:
    f.write("""
## Stronger insulin probe rejected; Python 80, forwards 10–11
- beta=-24 passed factual check on first 100 tuning stays: delta NLL +0.00170.
- Rollouts failed to improve doubling: -1.847 -> -1.854; halving .777 -> 1.334.
- NE doubling -.498 -> -.616; halving -.600 -> -.638.
- Thus rollout leverage is strongly nonlinear; do not extrapolate beta=-2 response.
- Saved beta=-24 checkpoint; restored the beta=-2 checkpoint pending diagnosis.
""")
print("Restored beta=-2 checkpoint; current edits:", edited())
```
