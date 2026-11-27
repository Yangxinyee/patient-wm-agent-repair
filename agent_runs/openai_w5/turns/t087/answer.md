The randomness test stopped on a broadcasting error, so it produced no evidence about the candidate. No weights changed. I’ll flatten the time-gap array and rerun the identical-state test. Differences under the old wrapper but not the corrected one would confirm that sampling noise contaminated the earlier sensitivities.

```python
import numpy as np
import torch

def next_means_crn74(j, h_branches, nsamples=16, seed=913):
    rr = np.flatnonzero(np.asarray(cacheTrain["previous"]) == int(j))
    if not len(rr):
        return None, None
    n = len(cacheTrain["df"])
    base_cache = {}
    for key, value in cacheTrain.items():
        if isinstance(value, pd.DataFrame):
            base_cache[key] = value.iloc[rr].reset_index(drop=True)
        elif isinstance(value, (np.ndarray, torch.Tensor)) and len(value) == n:
            base_cache[key] = value[rr]
        else:
            base_cache[key] = value
    dt_local = torch.as_tensor(
        base_cache["dt"], dtype=torch.float32).reshape(-1)
    outputs = []
    for branch in h_branches:
        cc = dict(base_cache)
        hh_local = branch[None].expand(len(rr), 4, 192)
        flowed_local = hh_local * torch.exp(
            -dt_local[:, None, None] * torch.exp(W0["core.flow.a"]))
        cc["base"] = (wm_norm(hh_local.mean(1), "core.out_norm")
                      + wm_norm(flowed_local.mean(1), "core.out_norm"))
        result = head_features_layout(
            cc, layout=head_layout_spec["layout"],
            ageunit=head_layout_spec["ageunit"],
            nsamples=nsamples, seed=seed)
        outputs.append(np.asarray(as_np(result["means"])).reshape(-1))
    return np.stack(outputs), cacheTrain["df"].iloc[rr]["event"].to_numpy()

row75 = int(sample68[np.flatnonzero(valid_jac73)[0]])
hsaved75 = torch.from_numpy(s4["x"][row75, 192:960]).reshape(1, 4, 192)
identical75 = hsaved75.expand(3, 4, 192).clone()
old_identical75, events75 = next_means61(row75, identical75)
new_identical75, _ = next_means_crn74(row75, identical75)

print("Time gap shape:", np.asarray(cacheTrain["dt"]).shape)
print("Flow parameter shape:", tuple(W0["core.flow.a"].shape))
print("Events:", events75)
print("Old readout, identical states:\n", old_identical75)
print("Corrected readout, identical states:\n", new_identical75)
print("Old maximum difference:",
      np.max(np.abs(old_identical75 - old_identical75[0])))
print("Corrected maximum difference:",
      np.max(np.abs(new_identical75 - new_identical75[0])))
print("Saved h RMS:", np.sqrt(np.mean(s4["x"][:, 192:960] ** 2)))

print("\n=== HEAD SAMPLING IMPLEMENTATION ===")
for line75 in compact_code73(head_features_layout).splitlines():
    if any(word in line75.lower() for word in
           ("rand", "sample", "seed", "z_prior", "logsd", "clamp")):
        print(line75)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 74–75: response randomness audit\n"
            "- Run 74 stopped on time-gap broadcasting; no edits or conclusions.\n"
            "- Run 75 corrects the gap shape and repeats the identical-state test.\n")
print("Current edits:", edited())
```