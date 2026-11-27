The blend predicts −13.15 and +5.54 for insulin, and +5.73 and −2.62 for norepinephrine. Relative to the fallback, insulin improves modestly, while norepinephrine half-dose worsens slightly. I’ll check the paired differences and recover the decoder diagnostics before deciding whether a final adjustment is justified. A negligible or noisy insulin gain would favor the safer fallback.

```python
dose_o235 = pd.read_csv("/work/fwd_43/dose_edit_original.csv")
dose_e235 = pd.read_csv("/work/fwd_43/dose_edit_edited.csv")
keys235 = ["stay_id", "hour", "edit"]
compare235 = dose_edit234.merge(
    dose_e235, on=keys235, suffixes=("_fallback", "_blend"), validate="one_to_one"
)
assert np.array_equal(compare235["reached_fallback"], compare235["reached_blend"])
compare235 = compare235[compare235["reached_blend"].astype(bool)].copy()
compare235["change"] = compare235["shift_blend"] - compare235["shift_fallback"]

rows235 = []
for edit_name, group in compare235.groupby("edit"):
    clusters = group.groupby("stay_id")["change"].agg(["sum", "size"])
    mean = float(group["change"].mean())
    n = len(clusters)
    se = float(np.sqrt(n / (n - 1) *
               np.square(clusters["sum"] - mean * clusters["size"]).sum()) / len(group))
    rows235.append({
        "edit": edit_name,
        "fallback": float(group["shift_fallback"].mean()),
        "blend": float(group["shift_blend"].mean()),
        "difference": mean, "paired_stay_se": se,
        "equal_stay_blend": float(group.groupby("stay_id")["shift_blend"].mean().mean())
    })
comparison235 = pd.DataFrame(rows235)
print("PAIRED ROLLOUT COMPARISON")
print(comparison235.to_string(index=False))
original_check235 = dose_orig234.merge(dose_o235, on=keys235,
                                       suffixes=("_38", "_43"), validate="one_to_one")
print("Max original shift difference:",
      float(np.abs(original_check235["shift_38"] - original_check235["shift_43"]).max()))

print("RECOVERABLE DECODER / TARGET OBJECTS")
inventory235 = []
for name in sorted(list(globals())):
    if (any(t in name.lower() for t in ("rho", "target", "half", "blend", "causal"))
        or any(t in name for t in ("212", "218", "226", "227", "231", "232"))):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series, dict, list, tuple, float, int)):
            desc = (str(getattr(obj, "shape", "")) if hasattr(obj, "shape")
                    else str(len(obj)) if isinstance(obj, (dict, list, tuple))
                    else repr(obj))
            inventory235.append(f"{name}: {type(obj).__name__} {desc}")
print("\n".join(inventory235)[:5200])

print("EVIDENCE SECTION HEADINGS")
print("\n".join(s.splitlines()[0] for s in sections234[1:]
                if any(t in s.lower() for t in
                       ("instrument", "identif", "persistence", "norepinephrine", "placebo")))[:2200])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward43 / Run235: half-dose-blend rollout comparison\n")
    f.write(comparison235.to_json(orient="records") + "\n")
    f.write("Current candidate_B232 remains provisional; no weights changed. "
            "Factual cost +0.00347139. Insulin benefit is modest; NE half worsens slightly.\n")
```
