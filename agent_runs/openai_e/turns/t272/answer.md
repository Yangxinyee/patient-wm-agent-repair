The later working targets are roughly −13.6/+6.8 for insulin and +4.2–4.6/−2.1–2.3 for norepinephrine. The insulin-only edit already exceeds the factual budget. I’ll compare archived candidates using exact NLL and stay-averaged rollout shifts. Better half-dose correction at similar factual cost would favor reshaping the edit over simple shrinkage.

```python
print("EXACT FORWARD FRONTIER")
frontier235 = []
for directory235 in sorted(Path("/work").glob("fwd_*"),
                           key=lambda p: int(p.name.split("_")[-1])):
    row235 = {"forward": int(directory235.name.split("_")[-1])}
    for model235 in ["original", "edited"]:
        path235 = directory235 / ("factual_" + model235 + ".csv")
        if path235.exists():
            df235 = pd.read_csv(path235)
            nllcols235 = [c for c in df235.columns
                          if "nll" in c.lower() and pd.api.types.is_numeric_dtype(df235[c])]
            if nllcols235:
                col235 = nllcols235[0]
                row235[model235 + "_nll"] = float(df235[col235].mean())
                row235["n_values"] = len(df235)
            else:
                print("Unrecognized factual schema", directory235.name, df235.columns.tolist())
        path235 = directory235 / ("dose_edit_" + model235 + ".csv")
        if path235.exists():
            df235 = pd.read_csv(path235)
            reached235 = df235["reached"]
            if reached235.dtype != bool:
                reached235 = reached235.astype(str).str.lower().eq("true")
            use235 = df235.loc[reached235]
            staymeans235 = use235.groupby(["edit", "stay_id"])["shift"].mean()
            row235[model235 + "_dose_staymean"] = {
                k: round(float(v), 4)
                for k, v in staymeans235.groupby(level="edit").mean().items()}
            row235[model235 + "_dose_pooled"] = {
                k: round(float(v), 4)
                for k, v in use235.groupby("edit")["shift"].mean().items()}
    if "edited_nll" in row235 and "original_nll" in row235:
        row235["nll_cost"] = row235["edited_nll"] - row235["original_nll"]
    if len(row235) > 1:
        frontier235.append(row235)
        print(row235)

print("\nINSULIN CANDIDATE HISTORY")
budget235 = 2300
for sec235 in sections233:
    head235 = sec235.splitlines()[0] if sec235.splitlines() else ""
    nums235 = {int(x) for x in re.findall(r"\b(?:Run|Runs)\s+(\d+)", head235)}
    if nums235 & {58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 77, 200}:
        text235 = sec235[:min(310, budget235)]
        if budget235 > 0:
            print(text235)
            budget235 -= len(text235)

print("\nARCHIVE STRUCTURE AND NORMS")
baseline_delta235 = np.load("/work/baseline77.npz")["delta"].copy()
trial_delta235 = np.load("/work/insulin_trial219.npz")["delta"].copy()
for file235 in ["insulin_candidate58.npz", "insulin_candidate63.npz",
                "insulin_trial219.npz", "ne_candidate151.npz",
                "ne_candidate199.npz"]:
    with np.load("/work/" + file235, allow_pickle=False) as a235:
        print(file235, {k: a235[k].shape for k in a235.files})
        for k235 in ["delta", "ne_delta"]:
            if k235 in a235:
                value235 = a235[k235].astype(np.float64)
                sv235 = np.linalg.svd(value235, compute_uv=False)
                print(k235, "norm", round(float(np.linalg.norm(value235)), 3),
                      "top singular values", np.round(sv235[:4], 3))
print("baseline/trial difference norm:",
      float(np.linalg.norm(trial_delta235 - baseline_delta235)))

with Path("/work/notes.md").open("a") as f235:
    f235.write(
        "\n## Run 235: exact measured frontier audit\n"
        "Working early-tuning reached-hour targets: insulin about -13.6/+6.8; "
        "NE about +4.2..4.6/-2.1..2.3. Parsed archived paired factual costs and "
        "both stay-averaged and pooled dose shifts, and inspected insulin balancing "
        "candidate contracts. No weights changed. Python runs: 235; forwards: 27.\n"
    )
```