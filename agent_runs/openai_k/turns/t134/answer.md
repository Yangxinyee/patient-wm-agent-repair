The candidate diagnostics show modest intended responses and an unwanted norepinephrine effect on glucose, so they do not yet justify a repair. I’ll submit the moderate candidate for an official factual test, followed by a rollout test. If factual fit holds but the rollout remains weak or crosses between systems, I’ll reject this calibration and change the routing.

```python
import pandas as pd
candidate125 = candidates90[1.0]
assert not edited(), "Unexpected active edits; inspect before replacing."
for name125, tensor125 in candidate125.items():
    assert name125 in NAMES, name125
    array125 = tensor125.detach().cpu().numpy() if isinstance(tensor125, torch.Tensor) else np.asarray(tensor125)
    assert tuple(array125.shape) == tuple(SHAPES[name125])
    assert np.isfinite(array125).all()
    save_edit(name125, array125.astype(np.float32))
tuning_ids125 = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist())[:100]
print("Edited tensors:", edited())
print("Tuning IDs:", tuning_ids125)
for name125 in edited():
    print(name125, "relative change:",
          float(np.linalg.norm(load(name125) - load_original(name125)) /
                max(np.linalg.norm(load_original(name125)), 1e-12)))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 125: moderate token candidate submitted for testing\n"
            "- Saved candidates90[1.0], six posterior/token tensors, as current edits.\n"
            "- Local diagnostics suggest small factual changes but weak insulin response and NE-to-glucose cross-effects; these are provisional.\n"
            "- Next: official tuning factual and paired rollouts; reject if fit fails or routing is inadequate.\n"
            "- Python runs: 125; forward passes: 4.\n")
```
