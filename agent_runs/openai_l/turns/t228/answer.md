The counterfactual cache still has an unverified feature convention; no weights changed. I’ll inspect its numerical values rather than guess. Before using it for a repair, the reconstructed case inputs must reproduce the cached hidden activations.

```python
print("=== NUMERICAL CASE AUDIT ===")
for key in (("insulin", 0.0), ("insulin", 2.0),
            ("norepinephrine", 2.0)):
    rows, aa, bb, dr = canon_cases200[key]
    ff = Xraw204[:, rows]
    print("\n", key, "rows", rows[:5], "dose change", np.quantile(dr, [0, .5, 1]))
    for label, vv in (("raw factual", ff), ("A", aa), ("B", bb)):
        print(label, "first", np.round(vv[0, 0, :8], 5),
              "last", np.round(vv[0, 0, -5:], 5),
              "last-col range", (float(vv[..., -1].min()), float(vv[..., -1].max())),
              "mean/SD", float(vv.mean()), float(vv.std()))
    print("RMS A-factual / B-factual / A-B:",
          *[float(np.sqrt(np.mean(z*z))) for z in (aa-ff, bb-ff, aa-bb)])
    for label, vv in (("A", aa), ("B", bb)):
        tests = {}
        if len(mu192) == 192:
            for mode in ("unstandardize", "standardize"):
                cc = vv.copy()
                if mode == "unstandardize":
                    cc[..., :192] = cc[..., :192]*sd192 + mu192
                else:
                    cc[..., :192] = (cc[..., :192]-mu192)/sd192
                tests[mode] = round(float(np.sqrt(np.mean((cc-ff)**2))), 6)
        print(label, "transform RMS to factual", tests)

print("\n=== RECENT SOURCE-LIKE STRINGS ===")
for k in ("text198", "notes181", "notes182"):
    if k in globals():
        v = globals()[k]
        print(k, type(v).__name__, repr(v)[:1800])
print("\n=== CAPTURED CASE INPUT ===")
print("case180 delta quantiles", np.quantile(case180["delta_rate"], [0, .5, 1]))
print("case180 input first/last", case180["input"][0, 0, :8],
      case180["input"][0, 0, -5:])
print("\nNo weights changed.")
```