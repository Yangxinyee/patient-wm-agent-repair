The architecture confirms that the current edit changes only value predictions, leaving the other heads directly untouched. The recovery output was truncated by large function defaults. I’ll use compact diagnostics to recover the causal evidence, aggregate factual checks, and nonlinear candidates; stronger local sensitivity alone will not justify a new edit.

```python
import re

print("=== EARLY EVIDENCE NOTES ===")
notes254 = Path("/work/notes.md").read_text()
print(notes254[:3300])

print("\n=== HELPERS: PARAMETER NAMES ONLY ===")
helper_names254 = []
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            params = list(inspect.signature(obj).parameters)
        except Exception:
            params = []
        helper_names254.append(name)
        print(name, params)

print("\n=== KEY OBJECT NAMES ===")
for name, obj in list(globals().items()):
    if any(s in name.lower() for s in
           ("nonlin", "hinge", "candidate", "target", "contrast",
            "frontier", "calibr", "coef")):
        if isinstance(obj, np.ndarray):
            print(name, obj.shape)
        elif isinstance(obj, pd.DataFrame):
            print(name, obj.shape)
        elif isinstance(obj, dict):
            print(name, "keys:", list(obj)[:8])

print("\n=== NONLINEAR CANDIDATES ===")
nf254 = pd.read_csv("/work/map_nonlinear_frontier.csv")
print("Columns:", list(nf254.columns))
print(nf254.head(5).to_string(index=False))

print("\n=== CURRENT-EDIT FACTUAL BATCHES ===")
for k in [29, 30, 31, 32]:
    po = Path(f"/work/fwd_{k}/factual_original.csv")
    pe = Path(f"/work/fwd_{k}/factual_edited.csv")
    if po.exists() and pe.exists():
        oo, ee = pd.read_csv(po), pd.read_csv(pe)
        if k == 29:
            print("Columns:", list(ee.columns))
        sc = next((c for c in ee.columns if "stay" in c.lower()), None)
        nc = [c for c in ee.columns if "nll" in c.lower()]
        print("Directory", k, "rows", len(ee),
              "stay range", (ee[sc].min(), ee[sc].max()) if sc else None,
              "NLL deltas", {c: float(ee[c].mean()-oo[c].mean()) for c in nc})

print("\n=== FIRST TUNING BATCH IDS ===")
th254 = pd.read_csv("/data/tuning.csv", nrows=1)
sidcol254 = next(c for c in th254.columns if "stay" in c.lower())
tuning_ids254 = pd.read_csv("/data/tuning.csv", usecols=[sidcol254])[sidcol254].drop_duplicates().tolist()
print(tuning_ids254[:100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 254: compact recovery\n"
            "- No weights changed. Avoided printing large helper defaults.\n"
            "- Recovering causal evidence, current-edit factual batches, and nonlinear frontier.\n")
```