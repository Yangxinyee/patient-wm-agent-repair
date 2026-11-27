The architecture confirms that doses enter a shared recurrent update, which explains why a norepinephrine edit can also disturb factual predictions elsewhere. The output was truncated before the calibrated targets and helpers appeared. I’ll retrieve those selectively; they should show whether a more MAP-specific correction is feasible.

```python
import re, inspect
_notes117 = Path("/work/notes.md").read_text()
_sections117 = re.split(r"(?=^## )", _notes117, flags=re.M)
print("=== CALIBRATION AND VALIDATION NOTES ===")
for _sec117 in _sections117:
    _title117 = _sec117.splitlines()[0] if _sec117 else ""
    if re.search(r"(Run[s]? (?:20|21|22|57|58|59|60|63|64|65|76|77|78|79|80|81|82|83|84|85)\b|Forward (?:8|9|10|11|12)\b)", _title117):
        print(_sec117[:650])
print("=== FUNCTIONS ===")
_funcs117 = []
for _name117, _obj117 in list(globals().items()):
    if not _name117.startswith("_") and callable(_obj117) and hasattr(_obj117, "__code__"):
        _args117 = _obj117.__code__.co_varnames[:_obj117.__code__.co_argcount]
        _funcs117.append(_name117 + str(_args117))
print("\n".join(_funcs117)[:3500])
print("=== RECENT AND CORE OBJECTS ===")
for _name117, _obj117 in list(globals().items()):
    if _name117.startswith("_"):
        continue
    if (re.search(r"(77|95|103|107|108|114|115)$", _name117)
        or _name117.lower() in ("theta", "panel", "p", "train", "tuning", "states", "weights", "w", "d", "betas", "targets")):
        if isinstance(_obj117, (np.ndarray, torch.Tensor)):
            print(_name117, tuple(_obj117.shape), str(_obj117.dtype))
        elif isinstance(_obj117, dict):
            print(_name117, "dict keys:", list(_obj117)[:12])
        elif isinstance(_obj117, (list, tuple)):
            print(_name117, type(_obj117).__name__, "length", len(_obj117))
        elif isinstance(_obj117, (str, int, float, bool, np.number)):
            print(_name117, str(_obj117)[:120])
print("=== SELECTED TENSOR SHAPES ===")
print({n: SHAPES[n] for n in NAMES if any(s in n for s in
      ("b_in", "c_out", "delta_in", "a_in", "attr_heads.value_bin"))})
with open("/work/notes.md", "a") as _f117:
    _f117.write("\n## Run 117: compact calibration/helper recovery\n"
                "Run116 architecture confirms a shared recurrent dose pathway; output was truncated before helper inventory. "
                "Retrieving only calibration/validation notes and recent objects to avoid repeated large dumps. "
                "No weights changed. Python runs: 117; forwards: 15.\n")
```