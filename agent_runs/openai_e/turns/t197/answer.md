The comparison is unusable because the metadata rows were in different orders—the hour mismatch reached 25 hours. That can explain the large input errors and NLL. I’ll align rows by stay, hour, and event, then repeat the checks. Near-zero input differences would confirm an ordering artifact rather than a decoder error.

```python
import numpy as np, pandas as pd, torch

keys168 = ["stay_id", "hour", "event"]
assert not tune_meta167.duplicated(keys168).any()
assert not head_meta.duplicated(keys168).any()

aligned_meta168 = (
    tune_meta167.set_index(keys168)
    .reindex(pd.MultiIndex.from_frame(head_meta[keys168]))
    .reset_index()
)
assert aligned_meta168.state_row.notna().all()
static168, transition168, register168 = build_static167(aligned_meta168)

print("ALIGNED STATIC / TRANSITION ERRORS")
for k in ["head_static", "static_all122"]:
    diff = static168 - globals()[k]
    print(k, "max", float(diff.abs().max()),
          "rms", float(diff.square().mean().sqrt()))
print("transition max", float((transition168 - head_transition).abs().max()))

gpu168 = (
    orig159.set_index(keys168)
    .reindex(pd.MultiIndex.from_frame(head_meta[keys168]))
    .reset_index()
)
b168 = aligned_meta168.target_bin.to_numpy(dtype=int)
nll168 = -np.log(np.maximum(p167[np.arange(len(b168)), b168], 1e-30))
print("LOCAL NLL", nll168.mean(), "GPU NLL", gpu168.nll.mean())
print("LOCAL/GPU MEAN RMSE",
      np.sqrt(np.mean((r160[0].numpy() - gpu168.predicted_mean.to_numpy())**2)))
print(pd.DataFrame({
    "event": aligned_meta168.event,
    "local_nll": nll168,
    "gpu_nll": gpu168.nll,
}).groupby("event").mean().to_string())

print("REGISTER MAX ERROR",
      float((register168 - torch.tensor(reg162)).abs().max()))
print("HEAD CENTRES FIRST TWO", head_centres[:2].numpy().tolist())

train_static168, train_transition168, train_register168 = build_static167(train_meta162)
train_centres168 = torch.tensor(np.asarray([
    [0.0] + VOCAB["value_bin_centres"][ev]
    for ev in train_meta162.event
], dtype=np.float32))
train_original168 = np.load("/work/fwd_22/states_original.npz")
train_edited168 = np.load("/work/fwd_22/states_edited.npz")
assert np.array_equal(train_original168["stay_id"], train_edited168["stay_id"])
assert np.array_equal(train_original168["hour"], train_edited168["hour"])
print("TRAIN INPUT SHAPES", train_static168.shape, train_transition168.shape,
      train_centres168.shape)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 168: correct calibration row ordering\n"
        "Run167 compared metadata in raw record order against GPU/head order; its "
        "large static errors and NLL were invalid. Reindexed by stay/hour/event, "
        "repeated exact static/transition checks, and built paired training inputs. "
        f"Aligned local NLL={nll168.mean():.6f}; GPU NLL={gpu168.nll.mean():.6f}. "
        "No weights changed. Python runs: 168; forwards: 22.\n"
    )
```