Prior averaging did not close the gap, so the register correction is only partial. The many changed bins may reflect an edge-equality convention rather than bad labels. I’ll test left- and right-inclusive binning explicitly. If neither matches the paired GPU predictions, I’ll obtain fresh states rather than trust these calculations.

```python
import numpy as np
import pandas as pd
import torch

bins233 = {}
for side in ["left", "right"]:
    observed = np.zeros(len(head_rows), dtype=np.int64)
    last = np.zeros(len(head_rows), dtype=np.int64)
    for ev in head_rows.event.unique():
        ix = np.flatnonzero(head_rows.event.to_numpy() == ev)
        edges = num_spec[(ev, "value")]["fine_bin_edges"]
        observed[ix] = np.searchsorted(edges, head_rows.value.iloc[ix].to_numpy(), side=side) + 1
        jx = ix[head_rows.last_value.iloc[ix].notna().to_numpy()]
        last[jx] = np.searchsorted(edges, head_rows.last_value.iloc[jx].to_numpy(), side=side) + 1
    bins233[side] = (observed, last)
    print(side, "outcome disagreement with old labels:", int(np.sum(observed != _labels218)))

gpu233 = {m: pd.read_csv(f"/work/fwd_20/factual_{m}.csv") for m in ["original", "edited"]}
joins233 = {m: head_rows[["stay_id", "hour", "event"]].assign(ix=np.arange(len(head_rows))).merge(
    gpu233[m], on=["stay_id", "hour", "event"], validate="one_to_one") for m in gpu233}
avg233 = {}
summary233 = []
for side, (observed, last) in bins233.items():
    head_vb = last
    gen = torch.Generator().manual_seed(233)
    means = {m: np.zeros((len(head_rows), 17), dtype=np.float64) for m in gpu233}
    for draw in range(4):
        eps = torch.randn((len(head_rows), 32), generator=gen)
        for sign in [1, -1]:
            for model, weights, sf in [("original", WO225, orig231), ("edited", W225, edit232)]:
                p, _, _, _ = head225(weights, sf, noise=sign * eps, status=None, typeidx=1)
                means[model] += p.numpy() / 8
    avg233[side] = means
    comp = {}
    for model, probs in means.items():
        join = joins233[model]
        ix = join.ix.to_numpy()
        nll = -np.log(np.maximum(probs[np.arange(len(probs)), observed], 1e-30))
        mean = (probs * centres231).sum(1)
        comp[model] = (nll[ix], mean[ix], join.nll.to_numpy(), join.predicted_mean.to_numpy())
        summary233.append({"side": side, "model": model,
                           "local_nll": nll[ix].mean(), "gpu_nll": join.nll.mean(),
                           "mean_rmse": np.sqrt(np.mean((mean[ix] - join.predicted_mean.to_numpy())**2))})
    o, e = comp["original"], comp["edited"]
    print(side, "paired NLL delta local/GPU", float(np.mean(e[0]-o[0])), float(np.mean(e[2]-o[2])),
          "paired mean-shift RMSE", float(np.sqrt(np.mean(((e[1]-o[1])-(e[3]-o[3]))**2))))
print(pd.DataFrame(summary233).to_string(index=False))
head_vb = bins233["left"][1].copy()

diff233 = np.flatnonzero(bins233["left"][0] != bins233["right"][0])[:6]
print("edge-equality examples:",
      head_rows.iloc[diff233][["event", "value"]].to_dict("records"))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 233: bin-boundary audit\n"
            "- Prior averaging did not validate the recovered local head.\n"
            "- The register-layout improvement is partial, not proof of exact reconstruction.\n"
            "- Testing edge equality explicitly; changed bin indices must not automatically be treated as corrections.\n"
            "- No weight edits.\n")
```
