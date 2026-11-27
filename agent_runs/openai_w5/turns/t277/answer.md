The stronger head fails the factual constraint on this cohort: its increase is 0.00536 nats. I’ll restore the previously validated head, then inspect the dose encoder and saved-state interfaces. The next hypothesis is dose leakage into centering; removing that leakage should improve intervention symmetry without sacrificing factual fit.

```python
install_strength238(1.0, 1.0)
print("Restored pre-strength-trial value-head output weights.")
print("CONFIG", VOCAB["config"])
print("=== DATA README ===")
print(Path("/data/README.md").read_text()[:3500])
print("=== FUNCTIONS ===")
for fn in [drug_atom205, set205, linear205, norm205]:
    print(fn.__name__, inspect.signature(fn))
    try:
        print(inspect.getsource(fn)[:2500])
    except Exception:
        print("names", fn.__code__.co_names)
        print("constants", repr(fn.__code__.co_consts)[:900])
        buf = io.StringIO()
        dis.dis(fn, file=buf)
        print(buf.getvalue()[:2400])
print("=== AUDITS ===")
for nm in ["head_audit", "history_audit", "encoder_audit"]:
    print(nm, repr(globals()[nm])[:1200])
print("STATE SCHEMA", [(k, S[k].shape) for k in S.files])
print("HEAD ROW COLUMNS", list(head_rows.columns))
print(head_rows.head(3).to_string(index=False))
print("CORE SHAPES",
      [(k, SHAPES[k]) for k in NAMES
       if k.startswith(("core.prelude.", "core.prelude_norm", "z_post.", "z_in."))])
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 22 / run 243: reject strength-only trial\n"
            "- Tuning A factual NLL increase +0.00536 exceeds the +0.005 ceiling.\n"
            "- Restored the pre-trial value-head output weights (glucose strength 1, NE baseline).\n"
            "- Investigating whether post-dose ordinary features leak dose into the centering regression.\n")
```
