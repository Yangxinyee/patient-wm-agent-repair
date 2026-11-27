The live kernel has independent factual caches and saved insulin-only and combined candidates. I’ll now identify how their replacement neurons encode the drug signals and locate the raw perturbation features. If those features reproduce the installed direction’s response, I can build a lower-variance replacement without relying on the disputed transform.

```python
import inspect

print("=== EVALUATOR SIGNATURES ===")
for name in ["c210_eval", "c211_logits", "r201_queries", "r201_fixed", "r145_head"]:
    f = globals()[name]
    print(name, inspect.signature(f))
    try:
        print(inspect.getsource(f)[:1400])
    except Exception:
        print("code names:", f.__code__.co_names, "locals:", f.__code__.co_varnames)

print("=== PROBE / TRANSFORM VARIABLES ===")
for k in sorted(list(globals())):
    if k.startswith(("r133", "r134", "r135", "r137", "r138", "r145",
                     "r147", "r152", "r177", "r223", "r225")):
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:8] if isinstance(v, dict) else "")
print("=== CHANGED HEAD ROWS ===")
wn = "attr_heads.value_bin.0.weight"
bn = "attr_heads.value_bin.0.bias"
on = "attr_heads.value_bin.2.weight"
r228_original = {n: load_original(n) for n in [wn, bn, on, "attr_heads.value_bin.2.bias"]}
for label in ["insulin_only", "combined_weak", "combined_strong"]:
    ws = c207_tensor_weights(c206_candidates[label])
    rows = torch.where((ws[wn] - r228_original[wn]).abs().amax(1) > 1e-6)[0]
    print(label, "changed input rows:", rows.tolist())
    for j in rows.tolist():
        print(" row", j, "norm", float(ws[wn][j].norm()), "bias", float(ws[bn][j]),
              "output norm", float(ws[on][:, j].norm()))
print("=== NOTES NEAR RECENT DIRECTION FIT ===")
notes228 = Path("/work/notes.md").read_text()
sections228 = notes228.split("\n## ")
for section in sections228:
    if section.startswith(("Run 170", "Run 171", "Run 174", "Run 175",
                           "Run 176", "Run 177", "Run 178", "Run 179")):
        print(section[:1000])
```
