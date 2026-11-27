The paired-change regressions predict held-out changes reasonably well, but their gradients do not transfer between perturbations. I won’t treat them as reliable edit directions. Next I’ll test slower time decay: drug information may fade between infusion bundles and readings. Stronger, correctly signed rollouts with safe factual NLL would support that hypothesis.

```python
replay_ops(repair_candidates["numeric_gain4"])
scale("core.flow.a", 1.15)
repair_candidates["flow_slow_115"] = freeze_ops()

print("Installed numeric dose/rate gain 4 plus core.flow.a scale 1.15.")
print("This slows coordinates whose learned log decay rate is negative;")
print("the factual and rollout tests will determine whether it helps.")
print(ops())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 60: time-flow pilot\n"
            "- Controlled head regressions fit their own perturbations (CV R2 0.77-0.87), "
            "but fail cross-perturbation transfer; they are not validated edit gradients.\n"
            "- Installed numeric gain 4 plus core.flow.a x1.15.\n"
            "- Hypothesis: excessive between-bundle decay suppresses drug information before readings.\n"
            "- Check factual NLL first, then dose effects if safe.\n"
            "- Forward passes used: 24.\n")
```