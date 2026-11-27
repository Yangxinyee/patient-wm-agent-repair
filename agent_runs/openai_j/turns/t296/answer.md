The evidence supports insulin more directly than norepinephrine: the latter’s kernel relies on measured-history adjustment, not a strong sensor-noise instrument. Posterior-only amplification was also ineffective. I’ll check whether nonlinear features already in the value head offer an independent correction; held-out factual loss and paired rollout improvement must both support it.

```python
print("=== CACHED INPUT STRUCTURES ===")
for name in ["wm_train145", "wm_tune145", "wm_cf146",
             "curve236", "head177_compressed"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj)[:12])
        if name == "wm_cf146":
            obj = obj[("norepinephrine", 2.0)]
        elif name == "curve236":
            obj = obj["train"]
        for key, val in list(obj.items())[:10]:
            if isinstance(val, dict):
                print(" ", key, "dict keys:", list(val)[:10])
            elif hasattr(val, "shape"):
                print(" ", key, tuple(val.shape))
            elif isinstance(val, (int, float, str)):
                print(" ", key, val)

print("\n=== CONTRAST HELPER ===")
brief_source255(contrast235, 2500)

print("\n=== EXACT HEAD EVALUATOR ===")
brief_source255(head177_eval, 2500)

print("\n=== EARLIER HEAD-CONTRAST CHECK ===")
hc258 = pd.read_csv("/work/head_contrast_cv.csv")
print("Columns:", list(hc258.columns))
print(hc258.head(4).to_string(index=False))

print("\n=== MASKS AND LOCAL FEATURE OBJECTS ===")
for name, obj in list(globals().items()):
    if ((name.endswith(("234", "235", "236", "206", "218")))
        or name in ["fitmask", "cvmask", "mapdir242", "cal242"]):
        if isinstance(obj, np.ndarray) or (
                hasattr(obj, "shape") and not callable(obj)
                and not isinstance(obj, type(np))):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys:", list(obj)[:5])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 258: independent nonlinear-feature feasibility\n"
            "- NE kernel target requires sequential-unconfoundedness assumptions; it is not observed intervention truth.\n"
            "- Prior posterior-only probe was ineffective and restored.\n"
            "- Inspecting existing value-head hidden features for an independent low-rank output correction.\n"
            "- No weights changed.\n")
```