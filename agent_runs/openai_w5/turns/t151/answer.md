The earlier checks found broadly consistent insulin slopes, but the multi-hour targets still depend on identification and response-form assumptions. A separate concern is that private memory coordinates decay rapidly, so the cached output may be carrying most of the dose signal. I’ll inspect the reading helpers and timing before amplifying anything; loss of the signal before prediction would refute this transport design.

```python
print("=== REMAINING EARLY CAUSAL CHECKS ===")
print(notes130[3100:6300])

print("=== KEY HELPER INTERFACES ===")
for name132 in [
    "reading_cache", "head_parts", "head_features_layout", "head_mc",
    "head_mc_layout", "reconstruct_head", "next_means_crn74",
    "norm0", "mlp0", "context_from_saved", "wm_step_saved",
    "correction_data", "mechanistic_shifts", "array129"
]:
    obj132 = globals().get(name132)
    if inspect.isfunction(obj132):
        print(name132, inspect.signature(obj132))
        print("  names:", obj132.__code__.co_names)
        print("  locals:", obj132.__code__.co_varnames[:25])

print("=== READING-CACHE SAMPLE ===")
print(cacheE129["df"].head(8).to_string(index=False))
print("base private-coordinate mean/std:",
      np.mean(cacheE129["base"][:, R125], axis=0),
      np.std(cacheE129["base"][:, R125], axis=0))
print("dt range:", np.min(cacheE129["dt"]), np.max(cacheE129["dt"]))
print("=== AVAILABLE STATE / CAUSAL OBJECT NAMES ===")
print(", ".join(sorted(
    k for k, v in list(globals().items())
    if not inspect.isfunction(v)
    and any(s in k.lower() for s in ["cache", "state", "panel", "target", "mech", "causal", "train_ids"])
)[:100]))
```
