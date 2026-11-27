The earlier record analyses support an insulin effect near −2 glucose units per rate unit and a positive norepinephrine effect on MAP. Insulin estimates also agree with a dose-policy discontinuity analysis and a pre-dose placebo check. I’ll recover the exact rollout request and lag estimates now; the scaling probe must move predictions toward those estimates, not merely preserve factual fit.

```python
import json

print("=== PAIRED ROLLOUT REQUEST ===")
print(json.dumps({
    "type": "dose_edit",
    "model": "both",
    "split": "tuning",
    "stays": [int(x) for x in probe_stays29[:100]]
}))

print("\n=== CONTROLLED IMPULSE ESTIMATES ===")
print(irf_table.to_string(index=False))
print("Impulse coefficients:", {
    k: np.asarray(v).round(4).tolist() for k, v in irf_coefficients.items()
})

print("\n=== ACTION / LATENT / PRELUDE TENSOR SHAPES ===")
for _name in NAMES:
    if any(_s in _name for _s in (
        "encode_action", "z_post", "z_in", "prelude", "z_head", "attr_heads.value"
    )):
        print(_name, SHAPES[_name])

print("\n=== KEY OBJECT DETAILS ===")
print("H columns:", H.columns.tolist())
print("head_rows columns:", head_rows.columns.tolist())
print("State-related variable names:", [
    n for n in globals()
    if not n.startswith("_") and any(s in n.lower() for s in ("state", "tune", "split"))
])
print("Numeric specifications:", numeric_specs)
print("Dose-model types:", {k: type(v).__name__ for k, v in dose_models.items()})

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 32; forward 5\n"
        "- Recovered record-based identification: insulin immediate slope near -2; "
        "RD estimates agree and pre-dose lab placebo is near zero.\n"
        "- Printing only the rollout request, lag estimates, and relevant tensor shapes "
        "to avoid output truncation.\n"
    )
```