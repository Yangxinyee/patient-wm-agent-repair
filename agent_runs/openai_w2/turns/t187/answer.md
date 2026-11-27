The clean insulin threshold check supports a negative next-hour effect, with no comparable jump in lab availability or past outcomes. Norepinephrine threshold estimates are weak and unstable, so they are not a reliable calibration source. I’ll inspect the richer history-adjusted estimates and persistence fits; consistency there would justify retaining a positive norepinephrine target.

```python
wanted163 = (
    "## Run 11:", "## Run 12:", "## Run 22:",
    "## Run 33:", "## Run 38:", "## Run 39:"
)
text163 = "\n\n".join(
    section for section in sections160
    if section.startswith(wanted163)
)
print("=== REMAINING IDENTIFICATION EVIDENCE ===")
print(text163[:7700])

print("\n=== CALIBRATION OBJECTS ===")
for name163 in [
    "validated_B151", "validated_W151", "validated_gains151",
    "V128", "Borig128", "fit157", "fit158", "fit_ne157", "fit_g158",
    "record_impulse_targets", "linear_rollout_targets"
]:
    if name163 not in globals():
        continue
    obj163 = globals()[name163]
    if isinstance(obj163, dict):
        print(name163, list(obj163.keys())[:10])
    elif isinstance(obj163, (np.ndarray, torch.Tensor)):
        print(name163, tuple(obj163.shape))
        if name163 == "validated_gains151":
            print(obj163)
    elif callable(obj163):
        print(name163, "function")
    else:
        print(name163, type(obj163).__name__)
```
