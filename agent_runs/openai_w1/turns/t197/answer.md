The architecture confirms that the posterior branch acts through the recurrent state, not directly on the reading heads. I’ll check its activation and dose contrasts before increasing its gain. A clear posterior contrast with little rollout effect would suggest weak transmission; an absent contrast would point to a decoder or gating problem instead.

```python
import inspect, dis, io
import pandas as pd

print("=== BRANCH IMPLEMENTATION ===")
try:
    print(inspect.getsource(rp_install))
except (OSError, TypeError):
    _buf178 = io.StringIO()
    dis.dis(rp_install, file=_buf178)
    print(_buf178.getvalue()[:15000])

print("\n=== BRANCH SPECIFICATIONS ===")
for _drug, _spec in rp_branch_specs.items():
    print(_drug)
    for _k, _v in _spec.items():
        if hasattr(_v, "shape"):
            _a = _v.detach().cpu().numpy() if torch.is_tensor(_v) else np.asarray(_v)
            print(" ", _k, _a.shape, "norm", float(np.linalg.norm(_a)),
                  "range", (float(_a.min()), float(_a.max())))
        else:
            print(" ", _k, repr(_v)[:500])

def _post178(x, original=False):
    xx = torch.as_tensor(x, dtype=torch.float32)
    if original:
        w0, b0, w2, b2 = rp_old_w0, rp_old_b0, rp_old_w2, rp_old_b2
    else:
        w0, b0, w2, b2 = [load(n) for n in
            ("z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias")]
    with torch.no_grad():
        hh = torch.nn.functional.gelu(
            xx @ torch.as_tensor(w0, dtype=torch.float32).T +
            torch.as_tensor(b0, dtype=torch.float32))
        return (hh @ torch.as_tensor(w2, dtype=torch.float32).T +
                torch.as_tensor(b2, dtype=torch.float32)).numpy()

_p178 = _post178(rp_X) - _post178(rp_X, True)
print("\n=== OFFLINE POSTERIOR CONTRASTS ===")
print("factual mean-coordinate RMS", float(np.sqrt(np.mean(_p178[:, :32] ** 2))))
for (_drug, _mult), _xx in rp_cf_input.items():
    _m = np.asarray(rp_cf_masks[_drug], dtype=bool)
    _d = (_post178(_xx) - _post178(_xx, True) - _p178)[_m, :32]
    print(_drug, _mult, "n", len(_d), "mean-vector norm",
          float(np.linalg.norm(_d.mean(0))), "RMS", float(np.sqrt(np.mean(_d ** 2))))

print("\n=== RECORD-EFFECT CACHES ===")
for _k, _v in sorted(list(globals().items())):
    if any(s in _k.lower() for s in ("kernel", "impulse", "target_shift", "causal_target")):
        if hasattr(_v, "shape"):
            print(_k, tuple(_v.shape), repr(_v)[:600] if np.prod(_v.shape) <= 36 else "")
        elif isinstance(_v, dict):
            print(_k, repr(_v)[:1000])
        elif isinstance(_v, (list, tuple, int, float)):
            print(_k, repr(_v)[:800])
print("\n=== DATAFRAME CACHES ===")
for _k, _v in sorted(list(globals().items())):
    if isinstance(_v, pd.DataFrame) and not _k.startswith("_"):
        print(_k, _v.shape, list(_v.columns)[:16])

_notes178 = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE NOTES ===")
print(_notes178[:10000])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 178\nInspecting posterior gates and counterfactual activation before gain changes; "
             "recovering record-derived targets and data caches. No weights changed. "
             "Counts: 178 Python runs, 13 forwards.\n")
```
