The reserved features decode dose innovations, and the prior head is more responsive than the posterior-only route. I’ll check units and sensitivities before fitting the bridge between them. Correct signs for both glucose and both MAP sensors would support using it; saturation or cross-drug responses would argue against it.

```python
print("PRIOR RESPONSE DIRECTIONS: mean units per rate unit")
for penalty, direction in D65_candidates.items():
    response79 = np.einsum("ncs,sj->ncj", arr71(J65), arr71(direction))
    print(penalty, np.mean(response79, axis=0).round(3).tolist())

print("\nCOUNTERFACTUAL DATA FIELDS")
for item in CF59[:2]:
    print({k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:90])
           for k, v in item.items()})

print("\nRATE TARGET UNITS")
for name in ("rate66", "rtrue66", "dr66"):
    a = arr71(globals()[name])
    print(name, "min", a.min(axis=0), "max", a.max(axis=0),
          "mean", a.mean(axis=0), "sd", a.std(axis=0))
print("Z66 factual alignment error:",
      np.max(np.abs(arr71(Z66[:len(rate66), :-1]) -
                    arr71(Z66[:len(rate66), :-1]))))
for penalty, coef in post66_rate_models.items():
    pred79 = arr71(Z66)[:len(rtrue66)] @ arr71(coef)
    print("residual decoder", penalty,
          "RMSE", np.sqrt(np.mean((pred79-arr71(rtrue66))**2,axis=0)))

print("\nRETAINED TARGET / EFFECT SUMMARIES")
lines79 = Path("/work/notes.md").read_text().splitlines()
chosen79 = [s for s in lines79
            if any(t in s.lower() for t in
                   ("aggregate", "true target", "target shift", "impulse", "rollout target",
                    "simulated effect", "persistence"))]
print("\n".join(chosen79)[:2500])

print("\nFACTUAL INDEX ALIGNMENT")
print("fidx59", fidx59[:8], "future_index59", future_index59[:8])
print("observed counts:", np.asarray(observed59).sum(axis=0))
print("candidate coefficient finite:",
      {k: bool(np.isfinite(arr71(v)).all()) for k,v in post66_rate_models.items()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 79: units and prior sensitivity check\n"
            "- Reserved posterior features encode centered dose innovations; candidate prior effects must be calibrated in physical rate units.\n"
            "- Checking four-sensor signs, counterfactual fields, and retained dynamic targets before the first external candidate test.\n"
            "- Forward passes: 4; Python runs: 79.\n")
```