The notebook contains several independent effect checks and a latent-state persistence fit. Some simpler persistence estimators were unstable, so I won’t rely on them alone. I’ll read the successful checks and recover the current calibration objects. If the evidence supports materially stronger norepinephrine rollouts, I’ll investigate a correction that stays within the factual-fit budget.

```python
wanted161 = [
    "## Run 5:", "## Run 6:", "## Run 10:", "## Run 11:",
    "## Run 12:", "## Run 22:", "## Run 33:", "## Run 38:",
    "## Run 39:"
]
text161 = "\n\n".join(
    s for s in sections160 if any(s.startswith(p) for p in wanted161)
)
print("=== RECORD-ONLY IDENTIFICATION AND PERSISTENCE ===")
print(text161[:7800])

print("\n=== CURRENT CALIBRATION OBJECTS ===")
for k in ["validated_B151", "validated_W151", "validated_gains151",
          "constraint_candidates147", "V128", "Borig128",
          "fit_ne157", "fit_g158", "fit157", "fit158",
          "rho_g158", "rho_ne157", "record_targets",
          "linear_rollout_targets", "record_impulse_targets"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys:", list(v.keys())[:12])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape), v if v.size <= 8 if isinstance(v, np.ndarray) else False else "")
    elif callable(v):
        print(k, "function")
    else:
        print(k, repr(v)[:200])
```
