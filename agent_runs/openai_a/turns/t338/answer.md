Across all 400 tuning stays, the edit increases factual NLL by 0.00406 nats, within the limit. The bootstrap interval extends above 0.005, so unseen-stay safety remains uncertain. I’ll verify the checkpoint and recover the record-derived targets before testing another rollout batch; consistent insulin improvements would support retaining this edit.

```python
def _recovery_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== CHECKPOINT INTEGRITY ===")
for _name in sharp240_checkpoint:
    _a = _recovery_np(load(_name))
    _b = _recovery_np(sharp240_checkpoint[_name])
    print(_name, "max difference", float(np.max(np.abs(_a - _b))))
    assert np.array_equal(_a, _b)
assert set(edited()) == set(sharp240_checkpoint)

print("=== KEY RECORD-DERIVED IDENTIFICATION RESULTS ===")
for _wanted in (
    "Recovery and threshold identification; Python 23",
    "Sustained insulin target and NE flexible adjustment; Python 24",
    "NE boundary uncertainty; Python 255",
):
    _matches = [s for s in _sections if _wanted in s.splitlines()[0]]
    for _s in _matches:
        print(_s[:2400])

print("=== OTHER NE IDENTIFICATION CONCLUSIONS ===")
_ne_lines = []
for _s in _sections:
    _ls = _s.splitlines()
    _relevant = [
        x for x in _ls if x.startswith("- ") and
        re.search(r"norepinephrine|\bNE\b", x) and
        re.search(r"effect|estimate|identif|IV|causal|placebo|weak|null|positive|negative|observational", x, re.I)
    ]
    if _relevant:
        _ne_lines.append(_ls[0])
        _ne_lines.extend(_relevant)
print("\n".join(_ne_lines)[-2300:])

print("=== TARGET AND IDENTIFICATION OBJECTS ===")
_obj_lines = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not re.search(
        r"(^|_)(beta|rho|target|sustained|iv|rd|effect|est)(_|[0-9]|$)|^targ", _name, re.I
    ):
        continue
    if callable(_obj):
        try:
            _desc = str(inspect.signature(_obj))
        except Exception:
            _desc = "callable"
    elif np.isscalar(_obj):
        _desc = repr(_obj)
    elif isinstance(_obj, tuple):
        _desc = repr([
            float(v) if np.isscalar(v) and isinstance(v, (int, float, np.number))
            else (type(v).__name__, getattr(v, "shape", None))
            for v in _obj
        ])
    else:
        _desc = f"{type(_obj).__name__}, shape={getattr(_obj, 'shape', None)}"
    _obj_lines.append(_name + " " + _desc)
print("\n".join(_obj_lines)[:1400])

_last100_ids = [int(x) for x in tuning_ids282[-100:]]
print("=== LAST100 IDS FOR INDEPENDENT ROLLOUT ===")
print(json.dumps(_last100_ids))
with open("/work/notes.md", "a") as _fh:
    _fh.write(
        "\n## Sharp240 integrity and safety uncertainty; Python 285, forwards 39/50\n"
        "- Current tensors equal sharp240_checkpoint exactly; only four value-head tensors edited.\n"
        "- Full400 factual delta +0.00405552; paired stay-bootstrap 95% interval "
        "[0.00262195, 0.00555593]. Tuning passes, but unseen-stay safety is not guaranteed.\n"
        "- Preparing independent last100 dose-edit validation; no weights changed.\n"
    )
```
