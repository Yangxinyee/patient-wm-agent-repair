This repeats the previous inspection: no weights are edited, and both numeric inputs and bin embeddings carry dose magnitude. It does not yet show normalization or feature timing. I’ll obtain those details before editing; post-treatment leakage in the statistical features would weaken the null-effect conclusion.

```python
import pandas as pd

print("=== NUMERIC NORMALIZATION ===")
print(VOCAB["numeric_note"])
drug_codes = {8, 9, "8", "9", "insulin", "norepinephrine"}
for row in VOCAB["numeric"]:
    if row["code"] in drug_codes:
        print(row)

print("\n=== PANEL AND PRETREATMENT FEATURES ===")
print("panel index:", panel.index.names)
print(panel.head(4).to_string())
print("history columns:", list(Hrich.columns))
print("eligibility counts:", {
    "eligible": int(np.sum(eligible)),
    "held_out": int(np.sum(held)),
    "fit_rows": int(np.sum(fit_rows)),
    "test_rows": int(np.sum(test_rows))
})
print("one-hour adjusted effects:", repr(strong_results)[:1800])
print("dose prediction parameters:", strong_params)

print("\n=== DRUG DISTRIBUTIONS ===")
for drug in DRUGS:
    rows = train.loc[train.event == drug]
    print(drug, "records:", len(rows))
    print(rows[["rate", "dose"]].describe(
        percentiles=[.05, .25, .5, .75, .95]
    ).to_string())

print("\n=== EMBEDDING INPUT WEIGHTS ===")
for name in ("embed.numeric.weight", "embed.bins.dose.weight",
             "embed.bins.rate.weight"):
    tensor = load_original(name)
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    print(name, "type:", type(tensor).__name__,
          "norm:", float(np.linalg.norm(arr)))
    if name == "embed.numeric.weight":
        print("column norms:", np.linalg.norm(arr, axis=0))

print("\n=== SAVED FORWARD OUTPUTS ===")
for directory in ("/work/fwd_1", "/work/fwd_2"):
    for path in Path(directory).iterdir():
        print(path.name)
        if path.suffix == ".csv":
            result = pd.read_csv(path)
            print("shape:", result.shape, "columns:", list(result.columns))
            print(result.head(2).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 9: normalization and identification audit\n"
            "- Checking dose/rate normalization and pretreatment feature definitions before an edit.\n"
            "- Candidate edit will collapse only positive dose/rate bins and numeric magnitude columns, preserving absent entries and presence flags.\n"
            "- Python runs: 9. Forward passes: 2. No edits yet.\n")
```