The run stopped at a normalization-key lookup, before constructing features or changing weights. The official factual table already contains the prior reading and its time, so I can avoid rebuilding those registers. I’ll correct the lookup and repeat the assay; close agreement would support reconstruction, while large errors would restrict these features to provisional proposals.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

print("Normalization top-level keys:", list(num_spec0.keys())[:20])

def r173_find_spec(ev):
    cid = code_id0[ev]
    nodes = [num_spec0]
    for _ in range(3):
        new = []
        for node in nodes:
            if isinstance(node, dict):
                if "median" in node and "iqr" in node:
                    return node
                for key in ["value", ev, cid, str(cid)]:
                    if key in node:
                        new.append(node[key])
            elif isinstance(node, (list, tuple)) and len(node) > cid:
                new.append(node[cid])
        nodes = new
    raise KeyError(("normalization not found", ev, cid))

r173_specs = {ev: r173_find_spec(ev) for ev in events151}
print("Resolved value normalization:", {
    ev: {k: spec[k] for k in ["median", "iqr"]}
    for ev, spec in r173_specs.items()
})
r173_fact = F143.reset_index(drop=True)
r173_ev = r173_fact.event.to_numpy()
r173_lv = r173_fact.last_value.to_numpy(float)
r173_age = np.where(
    np.isfinite(r173_fact.last_hour.to_numpy(float)),
    r173_fact.hour.to_numpy(float) - r173_fact.last_hour.to_numpy(float), 0.
)

def r173_head_features(query, event, last_value, age_hours,
                       value_col=2, flag_col=3, age_seconds=True,
                       activation="gelu"):
    query = torch.as_tensor(query, dtype=torch.float32)
    event = np.asarray(event)
    lv = np.asarray(last_value, dtype=float)
    ages = np.asarray(age_hours, dtype=float)
    n = len(event)
    codes = np.array([code_id0[e] for e in event], dtype=np.int64)
    bins = np.zeros(n, dtype=np.int64)
    z = np.zeros(n, dtype=np.float32)
    has = np.isfinite(lv)
    for ev in np.unique(event):
        ix = np.flatnonzero((event == ev) & has)
        spec = r173_specs[ev]
        bins[ix] = 1 + np.searchsorted(
            np.asarray(spec["fine_bin_edges"], dtype=float), lv[ix], side="left")
        z[ix] = np.clip((lv[ix] - spec["median"]) / spec["iqr"], -8, 8)
    bins = np.clip(bins, 0, 16)
    numeric = np.zeros((n, 8), dtype=np.float32)
    numeric[:, 0] = has
    numeric[:, 1] = np.log1p(np.maximum(ages, 0) *
                             (3600 if age_seconds else 1)) / 10
    numeric[:, value_col] = z
    numeric[:, flag_col] = has
    wc = (query + W0["embed.code.weight"][torch.as_tensor(codes)]
          + F.linear(torch.as_tensor(numeric),
                     W0["last_numeric.weight"], W0["last_numeric.bias"])
          + W0["embed.bins.value.weight"][torch.as_tensor(bins)]
          + W0["embed.bins.rate.weight"][0]
          + W0["last_status.weight"][0])
    pre = F.linear(wc, W0["attr_heads.value_bin.0.weight"],
                   W0["attr_heads.value_bin.0.bias"])
    hidden = F.gelu(pre) if activation == "gelu" else F.relu(pre)
    logits = (F.linear(hidden, W0["attr_heads.value_bin.2.weight"],
                       W0["attr_heads.value_bin.2.bias"])
              + W0["embed_transition.value_bin.weight"][
                  torch.as_tensor(codes * 17 + bins)])
    p = torch.softmax(logits, -1).numpy()
    centres = np.stack([
        np.r_[0., np.asarray(VOCAB["value_bin_centres"][e], dtype=float)]
        for e in event
    ])
    return {"hidden": hidden.numpy(), "p": p, "centres": centres,
            "means": (p * centres).sum(1), "codes": codes, "bins": bins}

r173_candidates, r173_results = {}, []
for vc, fc in [(2, 3), (6, 3), (6, 7)]:
    for sec in [True, False]:
        for activation in ["gelu", "relu"]:
            key = (vc, fc, sec, activation)
            result = r173_head_features(
                V144o["query"], r173_ev, r173_lv, r173_age,
                vc, fc, sec, activation)
            r173_candidates[key] = result
            mae = np.mean(np.abs(
                result["means"] - r173_fact.predicted_mean_o.to_numpy(float)))
            r173_results.append((key, float(mae)))
print("\nOriginal-mean MAE, best first:")
for key, mae in sorted(r173_results, key=lambda x: x[1]):
    print(key, round(mae, 4))

print("\nAction contrast helper dependencies:")
c = rr35_scaled_action_encoding.__code__
print("args", c.co_varnames[:c.co_argcount], "globals", c.co_names,
      "constants", [x for x in c.co_consts if isinstance(x, (str, int, float))])
for k in c.co_names:
    if k in globals() and not callable(globals()[k]) and hasattr(globals()[k], "shape"):
        print(k, tuple(globals()[k].shape))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 172–173: repair metadata lookup and head assay\n"
        "- Run 172 stopped at normalization lookup before feature construction "
        "or edits. Official F143 already supplies strict prior-reading registers.\n"
        "- Resolved normalization schema and assayed approximate head layouts "
        "against original official means. Reconstructions remain provisional "
        "unless agreement is close. Saved edits unchanged.\n"
        "- Python runs: 173; forward passes: 10.\n"
    )
```