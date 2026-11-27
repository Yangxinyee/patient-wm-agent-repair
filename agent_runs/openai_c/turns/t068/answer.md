The saved diagonal designs target the final state summary directly, unlike the latent-input edit that had no useful effect. Their size alone does not tell us whether they are safe. I’ll calculate their predicted dose responses and changes to factual reading means, then use those estimates to choose a small pilot. Forward results will decide whether the linear approximation is useful.

```python
print("DIAGONAL CONSTRAINT OUTPUTS")
for penalty, design_vec in diag_designs.items():
    dv = np.asarray(design_vec)
    stats_by_event = {}
    for event in ["glucose_finger", "map_arterial"]:
        hs = head_surrogates[event]
        OO = np.asarray(hs["O"])
        dm = ((OO - OO.mean(axis=0))*dv) @ hs["w"]
        stats_by_event[event] = {
            "mean": float(dm.mean()), "sd": float(dm.std()),
            "q95_abs": float(np.quantile(np.abs(dm), .95))}
    print(penalty, "response", (Bdiag@dv).tolist(),
          "max_coordinate", float(np.max(np.abs(dv))),
          "factual_linear_shift", stats_by_event)
print("\nSMALL CONSTRAINT VARIABLES")
for key in ["output_constraints", "constraints", "target", "z_reference"]:
    if key in globals():
        vv = globals()[key]
        if isinstance(vv, np.ndarray):
            print(key, vv.shape, vv.tolist() if vv.size<=20 else "large")
        elif isinstance(vv, dict):
            print(key, {str(k): summarize_saved(v) for k,v in vv.items()})
        else:
            print(key, type(vv).__name__)
print("\nNORMALIZATION ARRAYS")
for key in ["h_normalized", "out_states", "out_reconstructed", "out_mu",
            "norm_gamma_est", "norm_beta_est", "states_original", "paired_o"]:
    vv=globals().get(key)
    print(key, summarize_saved(vv) if not isinstance(vv, dict) else list(vv))
print("\nHEAD DATA AND BASELINE")
for event in ["glucose_finger", "map_arterial"]:
    print(event, head_surrogates[event]["frame"].head(2).to_dict("records"))
print("probe_ids", list(probe_ids))
print("baseline_nll", baseline_nll)
print("\nLATENT/FLOW TENSORS")
for n in NAMES:
    if any(s in n for s in ["z_prior", "z_in", "flow", "a_in", "delta_in"]):
        print(n, SHAPES[n])
```
