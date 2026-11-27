Removing action information increased value NLL by 0.01804 nats, well beyond the allowed 0.005. I’ll revert it. Next I’ll inspect the saved reconstruction and decoding diagnostics: a useful correction must respond to dose changes while remaining small on factual records, and held-out decoding tests will tell us whether the existing state supports that.

```python
import inspect, dis
for n in list(edited()):
    drop_edit(n)
print("Edits after reversion:", edited())

print("Selected tensor shapes:")
for n in NAMES:
    if n.startswith(("encode_action", "core.c_out", "core.b_in", "core.prelude",
                     "core.coda", "z_post", "z_prior", "z_head", "attr_heads.value_bin")):
        print(n, SHAPES[n])

print("Action norm statistics:")
for n in ["encode_action.norm.weight", "encode_action.norm.bias"]:
    a = load_original(n)
    print(n, "range", float(a.min()), float(a.max()),
          "rms", float(np.sqrt(np.mean(a*a))))
print("Saved diagnostic summaries:")
for key in ["head_layout_spec", "dyn_estimates", "dynamic_alternatives",
            "target_checks_A", "corrTrainA", "corrTrainB", "corrA"]:
    if key not in globals():
        continue
    obj_repair = globals()[key]
    if isinstance(obj_repair, dict):
        print(key, {str(k): (tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame))
                             else v if np.isscalar(v) or isinstance(v, str)
                             else type(v).__name__)
                    for k, v in obj_repair.items()})
print("Panel columns:", panel.columns.tolist(), "index:", panel.index.names)
print(panel.head(3).to_string())
print("Nuisance models:", [(k, type(v).__name__) for k, v in nuisance_models.items()])

for function_name in ["posterior_features4", "correction_data", "wm_step_saved"]:
    function_repair = globals()[function_name]
    print("\nFUNCTION", function_name)
    try:
        print(inspect.getsource(function_repair))
    except (OSError, TypeError):
        print(dis.Bytecode(function_repair).dis()[:6500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart, runs 49–51 / forward 7\n"
            "- Re-read architecture and recovered persistent objects.\n"
            "- Constant empty-action encoder fails factual constraint: tuning-A NLL "
            "1.65737 versus 1.63933 (+0.01804). Reverted both norm edits.\n"
            "- Need a dose-sensitive correction centered on factual treatment policy, "
            "not removal of all action information.\n")
```
