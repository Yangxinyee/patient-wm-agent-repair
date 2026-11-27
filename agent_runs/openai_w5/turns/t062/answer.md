The ablation was reverted successfully; only the diagnostic print failed because the loader returns Torch tensors. I’ll fix the conversion and inspect the recurrence and decoder checks. If dose information remains recoverable on held-out stays, a localized readout correction is worth testing; otherwise we need to repair how doses enter the state.

```python
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / forward 7\n"
            "- Empty-action ablation: factual NLL 1.65737 versus original 1.63933, "
            "+0.01804; fails the +0.005 constraint. Reverted all edits.\n")

print("Current edits:", edited())
for n in ["encode_action.norm.weight", "encode_action.norm.bias"]:
    a_repair = np.asarray(load_original(n))
    print(n, "range", float(a_repair.min()), float(a_repair.max()),
          "rms", float(np.sqrt(np.mean(a_repair*a_repair))))

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

for function_name in ["posterior_features4", "wm_step_saved"]:
    function_repair = globals()[function_name]
    print("\nFUNCTION", function_name, inspect.signature(function_repair))
    try:
        print(inspect.getsource(function_repair))
    except (OSError, TypeError):
        print(dis.Bytecode(function_repair).dis()[:4000])
```
