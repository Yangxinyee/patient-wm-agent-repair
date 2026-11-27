The kernel still contains the stochastic head reconstructions, local dose contrasts, and saved insulin fallback, so we can continue without rebuilding them. I’ll now recover the identification checks and candidate interfaces. The norepinephrine extension should follow the record-derived response, not merely strengthen its observational association.

```python
start168 = notes_text.index("## Factual baseline and latent dynamics")
end168 = notes_text.index("## Head reconstruction setup", start168)
print("=== RECORD-BASED CAUSAL EVIDENCE ===")
print(notes_text[start168:end168][:6200])

print("\n=== RELEVANT HELPER SIGNATURES ===")
for nm168 in ["wm_mc145", "wm_mc_raw146", "wm_probs151", "wm_loss151",
              "wm_rawloss152", "trial158_audit", "map164_mean",
              "repair_changed_action", "repair_core_rows", "kernel_readout"]:
    fn168 = globals()[nm168]
    print(nm168, inspect.signature(fn168), "uses:", fn168.__code__.co_names)

print("\n=== MC / SENSOR CANDIDATE STRUCTURES ===")
for nm168 in ["wm_train145", "wm_tune145", "wm_train146", "wm_tune146",
              "wm_cf146", "sensor165_fits", "sensor165_gates"]:
    if nm168 in globals():
        ob168 = globals()[nm168]
        if isinstance(ob168, dict):
            k168 = next(iter(ob168))
            v168 = ob168[k168]
            print(nm168, len(ob168), "first key", k168,
                  "value", {k: getattr(v, "shape", type(v).__name__)
                            for k, v in v168.items()} if isinstance(v168, dict)
                  else getattr(v168, "shape", type(v168).__name__))

with open("/work/notes.md", "a") as f168:
    f168.write("\n## Restart audit, runs 166–168\n"
               "- Re-read architecture: only the four value-head tensors are edited; all other heads and the recurrent core remain original.\n"
               "- Working insulin edit passed GPU factual checks on two independent groups of 100 tuning stays, delta +0.00307/+0.00270.\n"
               "- Sensor-specific MAP candidates and stochastic local reconstructions remain in memory. Recovering causal evidence and interfaces before selecting an extension.\n"
               "- Forward passes used: 17. No new edits installed.\n")
```
