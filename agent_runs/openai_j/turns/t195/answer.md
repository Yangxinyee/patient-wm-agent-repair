The record estimates support a positive norepinephrine effect, though its persistence is uncertain. I’ll first build a MAP-only addition that leaves the validated insulin correction intact. Low compression error and acceptable held-out loss would justify a GPU rollout test; otherwise I’ll discard it.

```python
import torch
import torch.nn.functional as F

print("=== RESPONSE-KERNEL EVIDENCE ===")
_note = Path("/work/notes.md").read_text()
print(_note[7051:10100])

print("=== MAP CANDIDATE ===")
for _key in [(1.0, 1.0), (3.0, 1.0), (0.3, 1.0)]:
    _v = map172_candidates[_key]
    print(_key, type(_v).__name__, getattr(_v, "shape", None),
          list(_v)[:8] if isinstance(_v, dict) else "")
print(map164_frontier.to_string(index=False)[:2100])

print("=== INSULIN CONSTRUCTION ===")
for _name in sorted(k for k in globals() if k.startswith("trial158_")):
    _v = globals()[_name]
    if hasattr(_v, "shape"):
        print(_name, tuple(_v.shape))
    elif isinstance(_v, (int, float, str)):
        print(_name, _v)

print("=== STOCHASTIC INPUTS ===")
for _label, _mc in [("train", wm_train145), ("tuning", wm_tune145)]:
    print(_label, [(k, tuple(v.shape)) for k, v in _mc.items()
                   if hasattr(v, "shape")])
print("=== GATE ALGEBRA ===")
for _j in trial158_free:
    _row = trial158_W0[int(_j)].numpy()
    _design = np.stack([trial158_coef[:192], trial158_gate[:192]], axis=1)
    _ab = np.linalg.lstsq(_design, _row, rcond=None)[0]
    print("unit", int(_j), "coef/gate", _ab,
          "reconstruction RMS", float(np.sqrt(np.mean((_design @ _ab - _row)**2))),
          "bias", float(trial158_b0[int(_j)]),
          "output_axis_RMS",
          float(np.std(trial158_W2[:, int(_j)].numpy() / np.where(wm_v151.numpy() != 0, wm_v151.numpy(), 1))))
print("gate bias", trial158_gate[-1], "correction bias", trial158_coef[-1])
```
