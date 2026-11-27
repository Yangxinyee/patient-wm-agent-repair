The regressions read the insulin direction negatively and the norepinephrine direction positively, but the two directions overlap strongly, and the low-variance insulin direction reverses sign. That makes a direct reading-head edit risky. I’ll instead test a fourfold amplification of numeric dose/rate inputs, leaving measurement inputs unchanged. It succeeds only if rollouts improve and factual NLL remains within budget.

```python
clear_ops()

numeric_dose_axis = np.zeros(6)
numeric_dose_axis[2] = 1 / np.sqrt(2)
numeric_dose_axis[4] = 1 / np.sqrt(2)

# Both drugs have equal normalized dose and rate scores.
# This multiplies that shared numeric signal by four, without changing
# value inputs, presence indicators, bins, or last-reading registers.
rank1("embed.numeric.weight", numeric_dose_axis, 3.0, side="in")

candidate_c_ops = list(ops())
saved_candidates["C_numeric_x4"] = candidate_c_ops

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 15: candidate C, action numeric amplification\n"
        "- Head surrogates give immediate insulin sensitivity -0.319 per unit "
        "and NE sensitivity +0.808 per rate unit, but are approximate.\n"
        "- Whitened directions overlap (cosine 0.848); whitening reverses "
        "the estimated insulin readout sign. Direct head amplification is not justified yet.\n"
        "- Applied one input-side rank-one edit to embed.numeric.weight: "
        "multiply the equal z_dose/z_rate signal by 4. Measurement numeric "
        "inputs, presence flags, bins, and reading registers are unchanged.\n"
        "- Test factual NLL first, then dose rollouts if the fit holds.\n"
        "- Python runs: 15; forward passes: 8.\n"
    )

print("Candidate C operations:", ops())
```