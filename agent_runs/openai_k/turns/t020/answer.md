The saved states include the previous recurrent state and posterior means, giving us a direct reconstruction check. I’ll inspect the helper implementations and record fields before testing more conventions. A localized posterior edit is worth considering only if we can reconstruct its inputs accurately and isolate dose changes from patient condition.

```python
import dis

print("=== PRIMITIVE HELPERS ===")
for name in ("lin0", "act0", "mlp0", "norm0"):
    fn = globals()[name]
    print("\n", name)
    try:
        print(inspect.getsource(fn))
    except Exception:
        print("names:", fn.__code__.co_names)
        print("constants:", fn.__code__.co_consts)
        dis.dis(fn)

print("\n=== RECORD SCHEMA AND EXAMPLES ===")
print(Path("/data/README.md").read_text()[:3500])
print(df.head(8).to_string(index=False))
print("\nPanel columns:", list(P.columns))
print("Panel index sample:", P.index[:4].tolist())
print("Rate columns:", list(rates.columns))
print("First single-action state indices:", first_drug0[:12])
print("History feature names:", feature_names)

print("\n=== POSTERIOR CANDIDATE DETAILS ===")
for drug in drug_codes:
    print(drug, sorted(posterior_candidates0[drug])[:3])

print("\n=== IMPULSE UNCERTAINTY OBJECTS ===")
for drug_outcome, result in dynamic_ir.items():
    print(drug_outcome, result if not isinstance(result, dict) else list(result))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 15: primitive and record-field inspection\n"
            "- Saved states provide direct posterior reconstruction targets and previous-state inputs.\n"
            "- Considering a conditionally centered posterior-mean edit, but only after validating reconstruction and signal isolation.\n"
            "- Python runs: 15; forward passes: 4; no weights edited.\n")
```