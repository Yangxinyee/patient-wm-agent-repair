The saved “head configuration” is not the most accurate one—it reports several units of prediction error—so I should not use it uncritically. I’ll compare the surviving decoder configurations and verify the recurrent reconstruction again. Close agreement with the tool would support using local perturbations; substantial disagreement would require fixing the reconstruction first.

```python
cfg_candidates109 = {}
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and "gap_mode" in obj and "type_id" in obj:
        cfg_candidates109[name] = obj.copy()
print("Decoder configurations:")
for name, obj in cfg_candidates109.items():
    print(name, obj)
print("\nRelevant notebook passages:")
note_lines109 = Path("/work/notes.md").read_text().splitlines()
for i, line in enumerate(note_lines109):
    if any(s in line.lower() for s in ["head reconstruction", "head-only", "decoder reconstruction", "last_numeric", "reconstruct the head"]):
        print("\n".join(note_lines109[max(0, i-1):min(len(note_lines109), i+4)]))
print("\nLocal decoder checks:")
decoder_checks109 = []
for name, conf in cfg_candidates109.items():
    try:
        result = local_value_head(linked_head108, conf, zmode="prior_mc", nsamples=64)
        score = {
            "name": name,
            "mean_rmse": float(np.sqrt(np.mean((np.asarray(result["mean"]) - linked_head108.predicted_mean.to_numpy())**2))),
            "nll_rmse": float(np.sqrt(np.mean((np.asarray(result["nll"]) - linked_head108.nll.to_numpy())**2)))
        }
        decoder_checks109.append(score)
        print(score)
    except Exception as exc:
        print(name, type(exc).__name__, str(exc)[:150])
print("\nCorrect-head recurrent reconstruction:")
mem109 = memory104(include_current=True)
h109, out109 = step104(repair_hflow101[core_pick104], repair_e101[core_pick104],
                      repair_u101[core_pick104], repair_z101[core_pick104],
                      mem109, heads=4, age_mode="both")
print("h RMSE:", float((h109 - sx_repair[core_rows104, 192:960].reshape(-1,4,192)).square().mean().sqrt()))
print("out RMSE:", float((out109 - sx_repair[core_rows104, :192]).square().mean().sqrt()))
```
