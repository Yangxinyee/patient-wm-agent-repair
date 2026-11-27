The weight edits were applied, but saving the candidate failed because the helper requires an argument. I’ll save the existing operations without applying them again, then check factual NLL. The time-decay hypothesis is still untested.

```python
current61 = ops()
print("Current operations:", current61)
assert len(current61) == 2
assert current61[-1]["name"] == "core.flow.a"

repair_candidates["flow_slow_115"] = freeze_ops(
    repair_candidates["numeric_gain4"] + [current61[-1]]
)
print("Saved flow_slow_115; operations were not reapplied.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Python runs 60-61: time-flow pilot installed\n"
            "- Run 60 applied numeric gain 4 plus core.flow.a x1.15, then failed only during candidate saving.\n"
            "- Run 61 saved the existing candidate without repeating the edits.\n"
            "- Controlled head regressions fit their own perturbations but fail cross-perturbation transfer.\n"
            "- Testing whether between-bundle decay suppresses drug information; factual NLL is untested.\n"
            "- Forward passes used: 24.\n")
```