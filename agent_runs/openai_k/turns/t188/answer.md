The old head reconstruction was not validated: its predicted means differ materially from the official tool’s. I won’t treat its NLL as authoritative. I’ll check a small set of input layouts against the paired factual results, then use any approximate features only for proposals that the forward tool must validate.

```python
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

r172_fact = F143.reset_index(drop=True).copy()
print("Official factual columns:", list(r172_fact.columns))
print("Normalization examples:", {
    e: num_spec0[e]["value"] for e in [events151[0], events151[2]]
})

r172_records = pd.read_csv("/data/tuning.csv")
r172_measurements = r172_records[r172_records.event.isin(events151)]
r172_lv = np.full(len(r172_fact), np.nan)
r172_age = np.zeros(len(r172_fact))
r172_sid = r172_fact["stay_id"].to_numpy()
r172_hour = r172_fact["hour"].to_numpy(dtype=float)
r172_ev = r172_fact["event"].to_numpy()
for (sid, ev), rows in r172_measurements.groupby(["stay_id", "event"]):
    ix = np.flatnonzero((r172_sid == sid) & (r172_ev == ev))
    if not len(ix):
        continue
    rows = rows.sort_values("hour")
    hh = rows.hour.to_numpy(float)
    vv = rows.value.to_numpy(float)
    pos = np.searchsorted(hh, r172_hour[ix], side="left") - 1
    good = pos >= 0
    r172_lv[ix[good]] = vv[pos[good]]
    r172_age[ix[good]] = r172_hour[ix[good]] - hh[pos[good]]

def r172_head_features(query, event, last_value, age_hours,
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
        spec = num_spec0[ev]["value"]
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
    means = (p * centres).sum(1)
    return {"hidden": hidden.numpy(), "p": p, "centres": centres,
            "means": means, "codes": codes, "bins": bins}

r172_mean_cols = [c for c in r172_fact.columns
                 if "mean" in c.lower() and pd.api.types.is_numeric_dtype(r172_fact[c])]
r172_candidates = {}
r172_results = []
for vc, fc in [(2, 3), (6, 3), (6, 7)]:
    for sec in [True, False]:
        for activation in ["gelu", "relu"]:
            key = (vc, fc, sec, activation)
            result = r172_head_features(
                V144o["query"], r172_ev, r172_lv, r172_age,
                vc, fc, sec, activation)
            r172_candidates[key] = result
            errors = {
                c: round(float(np.mean(np.abs(
                    result["means"] - r172_fact[c].to_numpy(float)))), 3)
                for c in r172_mean_cols
            }
            r172_results.append((key, errors))
print("\nApproximate-head mean MAE against official columns:")
for item in r172_results:
    print(item)

print("\nAction contrast helper dependencies:")
c = rr35_scaled_action_encoding.__code__
print("args", c.co_varnames[:c.co_argcount],
      "globals", c.co_names,
      "constants", [x for x in c.co_consts if isinstance(x, (str, int, float))])
for k in c.co_names:
    if k in globals() and not callable(globals()[k]) and hasattr(globals()[k], "shape"):
        print(k, tuple(globals()[k].shape))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 172: bounded head-input layout assay\n"
        "- Prior absolute-head reconstructions were inaccurate and are not "
        "authoritative NLL estimates.\n"
        "- Built reusable head-feature proposals under bounded register "
        "layouts and compared means against paired official factual outputs.\n"
        "- Any head edit must pass official factual and rollout tests. "
        "Saved edits unchanged. Python runs: 172; forward passes: 10.\n"
    )
```