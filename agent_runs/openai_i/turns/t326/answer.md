The architecture confirms that the edit changes the recurrent treatment pathway, but the output was truncated before the validation results appeared. I’ll extract just the recent losses, dose shifts, and causal estimates. This will show whether gain 0.8 is eligible to keep or needs further reduction.

```python
print("RECENT VALIDATION")
recent283 = {}
for k283 in [28, 29, 30, 31]:
    directory283 = f"/work/fwd_{k283}"
    if not os.path.isdir(directory283):
        continue
    for path283 in sorted(glob.glob(directory283 + "/*.csv")):
        df283 = pd.read_csv(path283)
        recent283[path283] = df283
        print("\n", path283, "rows", len(df283), "columns", df283.columns.tolist())
        if "factual" in path283:
            nc283 = [c for c in df283.columns if "nll" in c.lower()]
            print("NLL means",
                  df283[nc283].apply(pd.to_numeric, errors="coerce").mean().to_dict())
            sc283 = next((c for c in df283.columns if c in ["stay_id", "stay"]), None)
            if sc283:
                ids283 = df283[sc283].drop_duplicates().tolist()
                print("stay IDs", ids283)
        else:
            print("first row", df283.head(1).to_dict("records"))
            label283 = next(
                (c for c in df283.columns
                 if any(s in c.lower() for s in ["edit", "intervention", "scenario"])
                 and not pd.api.types.is_numeric_dtype(df283[c])), None)
            shift283 = [
                c for c in df283.columns
                if any(s in c.lower() for s in ["shift", "delta", "change"])
                and pd.api.types.is_numeric_dtype(df283[c])
            ]
            if label283 and shift283:
                print(df283.groupby(label283)[shift283].mean().to_string())

print("\nEXACT RECORD-TARGET NOTES")
lines283 = notes282.splitlines()
for i283, line283 in enumerate(lines283):
    if ("Targets are record-derived immediate slopes" in line283
        or "Earlier selected-subset convolution targets" in line283):
        print("\n".join(lines283[max(0, i283-1):i283+7]))

print("\nCURRENT RESPONSE GAIN")
with np.load("/work/neutral209.npz", allow_pickle=False) as neutral283, \
     np.load("/work/corrected269.npz", allow_pickle=False) as corrected283:
    print("checkpoint keys", corrected283.files)
    if "z_in.weight" in neutral283.files and "z_in.weight" in corrected283.files:
        base283 = neutral283["z_in.weight"].astype(np.float64)
        delta283 = corrected283["z_in.weight"].astype(np.float64) - base283
        current283 = as_numpy282(load("z_in.weight")).astype(np.float64)
        gain283 = float(np.sum((current283-base283)*delta283) / np.sum(delta283**2))
        print("gain", gain283, "max residual",
              float(np.max(np.abs(current283-base283-gain283*delta283))))

print("\nDATA SCHEMA")
print(open("/data/README.md").read()[:2200])

with open("/work/notes.md", "a") as log283:
    log283.write(
        "\n## Python 283 — compact validation recovery\n"
        "- Previous recovery output was truncated before its validation section.\n"
        "- Extracting recent factual NLLs, dose shifts, exact causal-target notes, and installed response gain.\n"
        "- No weights changed; gain-0.8 candidate remains provisional pending these results.\n"
    )
```