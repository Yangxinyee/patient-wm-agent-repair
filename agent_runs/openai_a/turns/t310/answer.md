Only two hidden neurons are currently replaced, leaving room for a separate glucose calibration term. I’ll compare the validated model’s GPU means with the centres of the observed glucose bins. A consistent signed error would justify testing a small intercept correction; little or inconsistent error would argue against it.

```python
import dis, io, inspect
print("=== GPU FACTUAL TABLES ===")
for _name in ("gpu400", "finger_gpu400", "gated8_gpu_tables"):
    _obj = globals().get(_name)
    print(_name, type(_obj).__name__)
    if hasattr(_obj, "columns"):
        print("shape", _obj.shape, "columns", list(_obj.columns))
        print(_obj.head(2).to_string(index=False))
    elif isinstance(_obj, dict):
        print("keys", list(_obj.keys())[:10])

print("=== SMALL GRAFT CONFIGURATION ===")
for _name in ("sc_names", "sc_nodes", "sc_rank", "sc_gate_b",
              "selective_sep_b", "activity_b"):
    print(_name, repr(globals().get(_name))[:280])
for _name in ("sc_gate_w", "selective_sep_w", "activity_w", "w_ins"):
    _obj = globals().get(_name)
    print(_name, "shape", np.shape(_obj),
          "norm", float(np.linalg.norm(_obj)) if _obj is not None else None)
for _name in ("probe_head_eval", "make_contrast_graft", "head_dose_data"):
    _obj = globals()[_name]
    print(_name, inspect.signature(_obj))
    print("global names:", _obj.__code__.co_names)
    print("constants:", repr(_obj.__code__.co_consts)[:350])
_buf = io.StringIO()
dis.dis(make_contrast_graft, file=_buf)
print("make_contrast_graft implementation:\n", _buf.getvalue()[:3000])

print("=== INSULIN EFFECT TARGETS ===")
for _name in ("rd_beta", "insulin_beta", "persistence", "insulin_persistence",
              "train_target", "train_keep", "train_e"):
    _obj = globals().get(_name)
    if isinstance(_obj, (int, float, np.number)):
        print(_name, _obj)
    elif hasattr(_obj, "shape"):
        print(_name, np.shape(_obj))

print("=== SELECTED NE NOTEBOOK SUMMARIES ===")
_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)(?=^## )", _notes)
_selected = [s for s in _sections if re.search(
    r"(?i)^## .*(?:NE |norepinephrine|flexible|DML|policy)", s)]
for _section in _selected[-5:]:
    print(_section[:650])
```
