The architecture confirms that the current edits target value predictions rather than the shared recurrent state. The early estimates also show substantial treatment-selection bias, so they cannot serve as causal targets alone. I’ll retrieve the later identification checks and candidate helpers with shorter output before testing another edit.

```python
import inspect, re

print("=== RECENT OBJECTS ===")
_r220_names = [
    _n for _n in list(globals())
    if re.search(r"(217|218|177|192|moment|pooled|head.*make|install)", _n, re.I)
]
for _n in sorted(_r220_names):
    _v = globals()[_n]
    if callable(_v):
        try:
            print(_n, "FUNCTION", inspect.signature(_v))
        except Exception:
            print(_n, "FUNCTION")
    elif isinstance(_v, dict):
        print(_n, "DICT", len(_v), "sample keys", list(_v)[:4])
    elif isinstance(_v, pd.DataFrame):
        print(_n, "FRAME", _v.shape, list(_v.columns))
    else:
        print(_n, type(_v).__name__, getattr(_v, "shape", None))

print("\n=== INSTALLER SOURCE ===")
if "head177_make" in globals():
    try:
        print(inspect.getsource(head177_make))
    except Exception as _e:
        print(type(_e).__name__, str(_e))

print("\n=== IDENTIFICATION NOTEBOOK EXCERPTS ===")
_r220_text = open("/work/notes.md").read()
_r220_sections = re.split(r"(?m)(?=^## )", _r220_text)
_r220_hits = [
    _s for _s in _r220_sections
    if re.search(r"(identification|placebo|instrument|kernel target|dynamic IV|sensor IV)", _s, re.I)
]
for _s in _r220_hits[-6:]:
    print(_s[:1000])
print("\n=== TARGET TABLE ===")
print(pd.read_csv("/work/kernel_targets.csv").head(12).to_string(index=False))
```