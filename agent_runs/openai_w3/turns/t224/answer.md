The architecture confirms that the dose signal enters the recurrent state through the posterior latent token. A carrier-only decoder can therefore be tested without changing the output heads. I’ll inspect the replay and compensation helpers first; the hypothesis would be weakened if isolated decoding still shifts factual states substantially.

```python
print("=== EXACT HELPERS ===")
for _name201 in ["latent187", "post187", "prelude155", "prelude_parts145",
                 "preludeE185"]:
    _fn201 = globals().get(_name201)
    if callable(_fn201):
        print("\n", _name201)
        try: print(inspect.getsource(_fn201))
        except Exception as _err201: print("source unavailable:", str(_err201))
print("\n=== CARRIER / CALIBRATION OBJECTS ===")
for _name201 in sorted(list(globals())):
    if (_name201.endswith("154") or _name201.endswith("187") or
        _name201.endswith("185") or _name201.endswith("186")):
        _obj201 = globals()[_name201]
        if isinstance(_obj201, dict):
            print(_name201, "dict", list(_obj201.keys())[:18])
        elif isinstance(_obj201, (np.ndarray, torch.Tensor)):
            print(_name201, type(_obj201).__name__, tuple(_obj201.shape))
        elif isinstance(_obj201, (list, tuple)):
            print(_name201, type(_obj201).__name__, "length", len(_obj201))
        elif callable(_obj201):
            try: print(_name201, inspect.signature(_obj201))
            except Exception: pass
print("\n=== PILOT AND RESTRICTED DECODER ===")
print("best192:", best192[:3])
print("pilot/base tensor RMS and max changes:")
for _name201 in pilot192:
    _a201 = np.asarray(pilot192[_name201])
    _b201 = np.asarray(base192[_name201])
    print(_name201, np.sqrt(np.mean((_a201-_b201)**2)), np.max(np.abs(_a201-_b201)))
print("restricted/full decoder inner products:")
print(restricted199.T @ restricted199)
print(restricted199.T @ decoder192)
print("primary calibration b193 norms:", np.linalg.norm(b193, axis=1))
```
