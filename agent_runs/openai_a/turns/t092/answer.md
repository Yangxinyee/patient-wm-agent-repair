The architecture confirms that changed doses reach predictions through the recurrent state. The recovered estimates consistently suggest a negative insulin effect, while norepinephrine estimates depend strongly on adjustment. I’ll recover the retained edit variables and stay IDs next, so the rollout test uses the same sample and a known edit strength.

```python
print("=== TUNING IDS, FIRST 100 ===")
print([int(x) for x in tuning_ids[:100]])
print("=== RETAINED FUNCTIONS ===")
print([k for k, v in list(globals().items())
       if callable(v) and not k.startswith("_")
       and getattr(v, "__module__", "") in {"__main__", None}])
print("=== EDIT / IDENTIFICATION VARIABLE NAMES ===")
_terms = ("graft", "robust", "safe", "rd", "beta", "decoder", "gate", "probe",
          "causal", "effect", "rho", "feature", "state", "head", "unit", "checkpoint")
_matches = [(k, v) for k, v in list(globals().items())
            if not k.startswith("_") and any(t in k.lower() for t in _terms)]
for k, v in _matches:
    if callable(v):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:12])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v))
    elif isinstance(v, (str, int, float, bool, type(None))):
        print(k, repr(v)[:150])
print("=== CAUSAL NOTEBOOK EXCERPTS ===")
_lines = open("/work/notes.md").read().splitlines()
for i, line in enumerate(_lines):
    if any(t in line.lower() for t in ["discontinu", "cutoff", "threshold", "persistence", "sustained", "rd beta"]):
        print("\n".join(_lines[max(0, i-1):min(len(_lines), i+3)])[:550])
print("=== CURRENT EDITS ===", edited())
```
