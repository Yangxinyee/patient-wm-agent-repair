The original rollout effects are small: about −1.6 for doubled insulin and +0.18 for doubled norepinephrine on the earlier sample. The adjusted record analyses suggest stronger responses, though those estimates still depend on adequate confounding control. I’ll check event timing and choose a tuning sample with enough exposed readings, so later comparisons are not dominated by stays where the drug never ran.

```python
print("=== ORIGINAL REACHED READOUTS ===")
for edit_name, grp in baseline_rollout.groupby("edit"):
    reached = grp[grp["reached"].astype(bool)]
    print(edit_name, "rows", len(reached), "stays", reached.stay_id.nunique(),
          "hour_mean", reached["shift"].mean(),
          "stay_mean", reached.groupby("stay_id")["shift"].mean().mean())

print("=== ACTUAL MEASUREMENT PANELS ===")
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    print(ev, P[ev].shape, P[ev][0, :8].tolist())
print("=== EXAMPLE RECORD TIMING ===")
example_sid = repair_state_ids_1[0]
print(train[train.stay_id == example_sid].head(20).to_string(index=False))

repair_S = dict(np.load("/work/fwd_4/states_original.npz"))
print("=== NEW STATES ===")
print({k: v.shape for k, v in repair_S.items()})
print(pd.DataFrame({k: repair_S[k][:14]
                    for k in ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]})
print("state rms:", np.sqrt(np.mean(repair_S["x"]**2, axis=0)).reshape(-1)[:8])
print("slot rms:", np.sqrt(np.mean(repair_S["x"][:,192:960]**2)))
print("=== FUNCTION AND PARAMETER DETAILS ===")
for fun_name in ["lin", "norm", "activation", "mlp", "act2", "repair_build_features"]:
    fun = globals().get(fun_name)
    if fun:
        print(fun_name, "names", fun.__code__.co_names,
              "constants", repr(fun.__code__.co_consts)[:800])
for name in ["core.coda.0.ffn.2.weight", "core.coda.0.ffn.0.weight",
             "core.coda.0.norm1.weight", "core.out_norm.weight", "z_in.weight"]:
    wt = load_original(name)
    print(name, "rms", float(np.sqrt(np.mean(wt**2))), "max", float(np.abs(wt).max()))

# Select a reproducible tuning set enriched for both relevant exposed readouts.
tc = tuning.groupby(["stay_id", "event"]).size().unstack(fill_value=0)
late = tuning[tuning.hour >= 12].groupby(["stay_id", "event"]).size().unstack(fill_value=0)
rng_repair = np.random.default_rng(9127)
ne_pool = late.index[(late.get("norepinephrine", 0) > 0) &
                    (late.get("map_arterial", 0) > 2)].to_numpy()
in_pool = late.index[(late.get("insulin", 0) > 0) &
                    (late.get("glucose_finger", 0) > 2)].to_numpy()
ne_selected = rng_repair.choice(ne_pool, min(50, len(ne_pool)), replace=False)
remaining_in = np.setdiff1d(in_pool, ne_selected)
in_selected = rng_repair.choice(remaining_in, min(100-len(ne_selected), len(remaining_in)), replace=False)
repair_eval_ids = [int(v) for v in np.r_[ne_selected, in_selected]]
print("NEW DOSE-EVAL IDS:", repair_eval_ids)
print("pool sizes:", len(ne_pool), len(in_pool), "selected:", len(repair_eval_ids))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 22\nChecked exposed-readout averaging and event timing; selected an enriched "
            "100-stay tuning intervention panel to measure both pathways with adequate support. "
            "Considering a low-rank coda correction decoded from dose innovations, rather than "
            "unconditional dose amplification. No edits.\n"
            "Counts: 22 Python runs, 4 forward passes.\n")
```
