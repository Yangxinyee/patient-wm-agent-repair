The run failed on an unclosed parenthesis, so it produced no evidence and changed no weights. I’ll rerun the corrected inspection. It should establish the timing and exposed-readout sample before we test a repair.

```python
print("=== ORIGINAL REACHED READOUTS ===")
for edit_name, grp in baseline_rollout.groupby("edit"):
    reached = grp[grp["reached"].astype(bool)]
    print(edit_name, "rows", len(reached), "stays", reached.stay_id.nunique(),
          "hour_mean", reached["shift"].mean(),
          "stay_mean", reached.groupby("stay_id")["shift"].mean().mean())

print("=== MEASUREMENT PANELS ===")
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    print(ev, P[ev].shape, P[ev][0, :8].tolist())
example_sid = repair_state_ids_1[0]
print("=== EXAMPLE TIMING ===")
print(train[train.stay_id == example_sid].head(16).to_string(index=False))

repair_S = dict(np.load("/work/fwd_4/states_original.npz"))
print("=== STATE LAYOUT ===")
print({k: v.shape for k, v in repair_S.items()})
print(pd.DataFrame({
    k: repair_S[k][:12]
    for k in ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]
}).to_string(index=False))
print("slot rms:", float(np.sqrt(np.mean(repair_S["x"][:, 192:960]**2))))

print("=== FUNCTION DETAILS ===")
for fun_name in ["norm", "activation", "mlp", "act2", "repair_build_features"]:
    fun = globals().get(fun_name)
    if fun:
        print(fun_name, fun.__code__.co_names, repr(fun.__code__.co_consts)[:650])
for name in ["core.coda.0.ffn.2.weight", "core.coda.0.ffn.0.weight",
             "core.coda.0.norm1.weight", "core.out_norm.weight", "z_in.weight"]:
    wt = np.asarray(load_original(name))
    print(name, "rms", float(np.sqrt(np.mean(wt**2))), "max", float(np.abs(wt).max()))

late = tuning[tuning.hour >= 12].groupby(["stay_id", "event"]).size().unstack(fill_value=0)
rng_repair = np.random.default_rng(9127)
ne_pool = late.index[(late["norepinephrine"] > 0) &
                    (late["map_arterial"] > 2)].to_numpy()
in_pool = late.index[(late["insulin"] > 0) &
                    (late["glucose_finger"] > 2)].to_numpy()
ne_selected = rng_repair.choice(ne_pool, min(50, len(ne_pool)), replace=False)
remaining_in = np.setdiff1d(in_pool, ne_selected)
in_selected = rng_repair.choice(
    remaining_in, min(100-len(ne_selected), len(remaining_in)), replace=False)
repair_eval_ids = [int(v) for v in np.r_[ne_selected, in_selected]]
print("NEW DOSE-EVAL IDS:", repair_eval_ids)
print("pool sizes:", len(ne_pool), len(in_pool), "selected:", len(repair_eval_ids))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 22–23\nRun 22 had a syntax error and executed nothing. "
            "Run 23 checks exposed-readout averaging and event timing and selects an enriched "
            "tuning intervention panel. No edits.\n"
            "Counts: 23 Python runs, 4 forward passes.\n")
```
