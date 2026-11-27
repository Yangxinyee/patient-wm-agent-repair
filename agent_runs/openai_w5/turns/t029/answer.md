The provisional hour-12 targets are about −11.9 and +5.9 for insulin, and +4.5 and −2.2 for norepinephrine. They remain estimates, not observed truth. I’ll now test the local head reconstruction and inspect its features. Small, unbiased reconstruction errors would support a head-only edit, which would leave the recurrent state and other heads unchanged.

```python
import torch
torch.set_num_threads(8)

def as_np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

cacheTrainB = reading_cache(train, np.load("/work/fwd_5/states_original.npz"))
headA0 = head_features_layout(cacheA, head_layout_spec["layout"],
                             ageunit=head_layout_spec["ageunit"], nsamples=8, seed=913)
headTrain0 = head_features_layout(cacheTrain, head_layout_spec["layout"],
                                 ageunit=head_layout_spec["ageunit"], nsamples=8, seed=913)
headTrainB0 = head_features_layout(cacheTrainB, head_layout_spec["layout"],
                                  ageunit=head_layout_spec["ageunit"], nsamples=8, seed=913)
for label, obj in [("A", headA0), ("train A", headTrain0), ("train B", headTrainB0)]:
    print(label, {k: getattr(v, "shape", None) for k, v in obj.items()})
means_local = as_np(headA0["means"]).reshape(-1)
nll_local = as_np(headA0["nll"]).reshape(-1)
for event in ["all", "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    keep = np.ones(len(cacheA["df"]), bool) if event == "all" else (
        cacheA["df"]["event"].to_numpy() == event)
    dm = means_local[keep] - cacheA["gpu_mean"][keep]
    dn = nll_local[keep] - cacheA["gpu_nll"][keep]
    print("RECONSTRUCTION", event, "n", int(keep.sum()),
          "mean bias/MAE", np.round([dm.mean(), np.abs(dm).mean()], 6),
          "NLL bias/MAE", np.round([dn.mean(), np.abs(dn).mean()], 6))
print("VOCAB KINDS", VOCAB["event_kinds"], VOCAB["atom_kinds"])
print("NORMALISATION", {str(k): (v["median"], v["iqr"])
                       for k, v in norm_meta.items()})
print("VALUE CENTRES", {k: np.round(v, 2).tolist()
      for k, v in VOCAB["value_bin_centres"].items()})
print("X INDEX/COLUMNS", X.index[:3].tolist(), list(X.columns))
print("WEIGHT TYPE", type(W0["attr_heads.value_bin.2.weight"]).__name__)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 23: head-only correction screening\n"
            "- Testing the selected register convention on original GPU predictions.\n"
            "- Preparing independent state samples for a no-gradient correction of treatment innovations.\n"
            "- Head-only edits would preserve the recurrent state and all other prediction heads directly.\n")
```
