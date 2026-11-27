Independent record-based checks support persistent drug effects much larger than the model’s current rollouts. Earlier edits to embeddings and latent-state updates barely changed those rollouts. I’ll inspect the saved readout designs and their factual-variance costs; a promising direction should increase dose sensitivity while changing ordinary predictions little.

```python
def summarize_saved(x, depth=0):
    if isinstance(x, dict):
        return {str(k): summarize_saved(v, depth+1) for k,v in x.items()}
    if isinstance(x, np.ndarray):
        return {"shape": x.shape, "norm": float(np.linalg.norm(x)),
                "values": x.tolist() if x.size <= 12 else None}
    if isinstance(x, pd.DataFrame):
        return {"shape": x.shape, "columns": list(x.columns)}
    if isinstance(x, (list, tuple)):
        return [summarize_saved(v, depth+1) for v in x] if len(x)<=12 else {"length":len(x)}
    if isinstance(x, (float,int,str,np.number)):
        return x
    return type(x).__name__

for key in ["diag_designs", "Bdiag", "Ddiag", "D_out", "G", "G_h",
            "state_dose_coefficients", "direction_diagnostics",
            "latent_candidate_basis", "latent_sensor_matrix", "latent_sv",
            "head_ne_design", "repair_candidates"]:
    if key in globals():
        print("\n"+key, summarize_saved(globals()[key]))
for event in ["glucose_finger", "map_arterial"]:
    hs = head_surrogates[event]
    print("\nHEAD", event, "w", summarize_saved(hs["w"]),
          "O", summarize_saved(hs["O"]), "R", summarize_saved(hs["R"]),
          "frame", summarize_saved(hs["frame"]))
print("\nNORMALIZATION ERROR",
      float(np.max(np.abs(out_reconstructed-out_states))) if np.shape(out_reconstructed)==np.shape(out_states) else "different saved shapes")
print("\nLATENT AND FLOW TENSORS")
for n in NAMES:
    if any(s in n for s in ["z_prior", "z_in", "flow", "a_in", "delta_in"]):
        print(n, SHAPES[n])
```
