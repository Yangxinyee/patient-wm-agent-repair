The wrapper uses the correct channels, but it evaluates branches through a sampled head. Different latent draws could masquerade as dose sensitivity. I’ll first test identical states, then give every branch the same latent draws. Nonzero differences between identical states would confirm this bug; identical outputs would refute it.

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
    dt_local = torch.as_tensor(base_cache["dt"], dtype=torch.float32)
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

row74 = int(sample68[np.flatnonzero(valid_jac73)[0]])
hsaved74 = torch.from_numpy(s4["x"][row74, 192:960]).reshape(1, 4, 192)
identical74 = hsaved74.expand(3, 4, 192).clone()
old_identical74, events74 = next_means61(row74, identical74)
new_identical74, _ = next_means_crn74(row74, identical74)

print("Events:", events74)
print("Old readout, three identical states:\n", old_identical74)
print("Common-random-number readout:\n", new_identical74)
print("Old identical-branch maximum difference:",
      np.max(np.abs(old_identical74 - old_identical74[0])))
print("Corrected identical-branch maximum difference:",
      np.max(np.abs(new_identical74 - new_identical74[0])))
print("Saved h RMS:",
      np.sqrt(np.mean(s4["x"][:, 192:960] ** 2)))

print("\n=== HEAD SAMPLING IMPLEMENTATION ===")
text74 = compact_code73(head_features_layout)
for line74 in text74.splitlines():
    if any(word in line74.lower() for word in
           ("rand", "sample", "seed", "z_prior", "exp", "logsd", "clamp")):
        print(line74)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 74: response randomness audit\n"
            "- Testing identical branches to detect latent-sampling noise in the old finite-difference readout.\n"
            "- Added a readout that evaluates each branch with the same latent random draws.\n"
            "- Existing response directions remain unvalidated; no edits made.\n")
print("Current edits:", edited())
```