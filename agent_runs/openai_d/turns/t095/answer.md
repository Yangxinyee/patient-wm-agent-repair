The rows are correctly aligned, but the cached local predictions still differ from the deterministic GPU predictions. This rules out row ordering as the explanation. I’ll test the time scaling, register layout, bin boundaries, and activation directly. A correct configuration should match both predicted means and reading NLLs to rounding precision.

```python
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

t85 = time.time()
torch.set_num_threads(8)
assert head_meta[["stay_id", "hour", "event"]].reset_index(drop=True).equals(
    probe84_df[["stay_id", "hour", "event"]].reset_index(drop=True)
)

R85 = {}
R85["W"] = {n: load_original(n).float() for n in NAMES}
R85["meta"] = head_meta.copy()
with np.load("/work/fwd_3/states_original.npz") as z85:
    states85 = torch.tensor(z85["x"][head_meta.state_idx.to_numpy()], dtype=torch.float32)

R85["out"] = states85[:, :192]
R85["h"] = states85[:, 192:960].reshape(-1, 4, 192)
R85["dt"] = torch.tensor(head_meta.dt.to_numpy(), dtype=torch.float32)
R85["has"] = torch.tensor(head_meta.has_last.to_numpy(), dtype=torch.float32)
R85["age"] = torch.tensor(head_meta.age.to_numpy(), dtype=torch.float32)
R85["last_z"] = torch.tensor(head_meta.last_z.to_numpy(), dtype=torch.float32)
R85["codes_map"] = {c["name"]: c["id"] for c in VOCAB["codes"]}
R85["code"] = torch.tensor([R85["codes_map"][e] for e in head_meta.event])
R85["numeric"] = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
R85["centres"] = torch.tensor(
    [[0.0] + VOCAB["value_bin_centres"][e] for e in head_meta.event],
    dtype=torch.float32
)
R85["scale"] = torch.tensor(
    [R85["numeric"][(e, "value")]["iqr"] for e in head_meta.event],
    dtype=torch.float32
)
R85["target_mean"] = torch.tensor(probe84_df.predicted_mean.to_numpy(), dtype=torch.float32)
R85["target_nll"] = torch.tensor(probe84_df.nll.to_numpy(), dtype=torch.float32)

def r85_linear(x, prefix):
    return F.linear(x, R85["W"][prefix + ".weight"], R85["W"][prefix + ".bias"])

def r85_norm(x, prefix):
    return F.layer_norm(
        x, (192,), R85["W"][prefix + ".weight"],
        R85["W"][prefix + ".bias"], 1e-5
    )

def r85_mlp(x, prefix, activation="gelu"):
    y = r85_linear(x, prefix + ".0")
    y = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[activation](y)
    return r85_linear(y, prefix + ".2")

def r85_age_feature(mode):
    if mode == "zero":
        return torch.zeros_like(R85["dt"])
    if mode.startswith("forecast"):
        age = R85["dt"]
        mask = 1.0
    elif mode.startswith("state"):
        age = (R85["age"] - R85["dt"]).clamp_min(0)
        mask = R85["has"]
    else:
        age = R85["age"]
        mask = R85["has"]
    if mode.endswith("hours"):
        age = age / 3600.0
    return mask * torch.log1p(age) / 10.0

def r85_bins(side):
    last, observed = [], []
    for row in head_meta.itertuples():
        info = R85["numeric"][(row.event, "value")]
        v = row.last_z * info["iqr"] + info["median"]
        last.append(0 if not row.has_last else
                    1 + np.searchsorted(info["fine_bin_edges"], v, side=side))
        observed.append(1 + np.searchsorted(
            info["fine_bin_edges"], row.observed, side=side))
    return torch.tensor(last), torch.tensor(observed)

R85["bins"] = {s: r85_bins(s) for s in ("left", "right")}
R85["gaps"] = {
    "seconds_div10": torch.log1p(R85["dt"])[:, None] / 10,
    "seconds": torch.log1p(R85["dt"])[:, None],
    "hours_div10": torch.log1p(R85["dt"] / 3600)[:, None] / 10,
    "hours": torch.log1p(R85["dt"] / 3600)[:, None],
    "log_seconds_div10": torch.log(R85["dt"].clamp_min(1))[:, None] / 10,
    "log_hours_div10": torch.log((R85["dt"] / 3600).clamp_min(1e-5))[:, None] / 10,
}
R85["flow_means"] = {
    "seconds": r85_norm(
        (R85["h"] * torch.exp(
            -R85["dt"][:, None, None] *
            torch.exp(R85["W"]["core.flow.a"])[None, None, :]
        )).mean(1), "core.out_norm"),
    "hours": r85_norm(
        (R85["h"] * torch.exp(
            -(R85["dt"] / 3600)[:, None, None] *
            torch.exp(R85["W"]["core.flow.a"])[None, None, :]
        )).mean(1), "core.out_norm"),
    "none": r85_norm(R85["h"].mean(1), "core.out_norm"),
}
print("State out reconstruction RMS:",
      float((R85["flow_means"]["none"] - R85["out"]).square().mean().sqrt()))

rng85 = np.random.default_rng(8501)
ix85 = torch.tensor(np.sort(rng85.choice(len(head_meta), 700, replace=False)))
B85 = R85["W"]["attr_heads.value_bin.0.weight"]
NB85 = B85 @ R85["W"]["last_numeric.weight"]
age_modes85 = [
    "register_seconds", "register_hours", "state_seconds",
    "state_hours", "forecast_seconds", "forecast_hours", "zero"
]
results85 = []

for flow_mode85, flow_mean85 in R85["flow_means"].items():
    for gap_mode85, gap_input85 in R85["gaps"].items():
        view85 = (R85["out"] + flow_mean85 +
                  r85_mlp(gap_input85, "gap") + R85["W"]["z_head.bias"])
        q85 = r85_norm(view85 + R85["W"]["type_emb.weight"][1], "query_norm")
        qpre85 = F.linear(q85[ix85], B85)
        for side85, (last85, observed85) in R85["bins"].items():
            fixed85 = (
                R85["W"]["embed.code.weight"][R85["code"]] +
                R85["W"]["embed.bins.value.weight"][last85] +
                R85["W"]["embed.bins.rate.weight"][0] +
                R85["W"]["last_status.weight"][0] +
                R85["W"]["last_numeric.bias"]
            )
            fixedpre85 = F.linear(
                fixed85[ix85], B85,
                R85["W"]["attr_heads.value_bin.0.bias"]
            )
            trans85 = R85["W"]["embed_transition.value_bin.weight"][
                R85["code"] * 17 + last85
            ][ix85]
            for zp85 in (2, 4, 6):
                for fp85 in (3, 5, 7):
                    for age_mode85 in age_modes85:
                        nf85 = torch.zeros(len(ix85), 8)
                        nf85[:, 0] = R85["has"][ix85]
                        nf85[:, 1] = r85_age_feature(age_mode85)[ix85]
                        nf85[:, zp85] = R85["last_z"][ix85]
                        nf85[:, fp85] = R85["has"][ix85]
                        pre85 = qpre85 + fixedpre85 + F.linear(nf85, NB85)
                        for activation85 in ("gelu", "silu"):
                            hidden85 = F.gelu(pre85) if activation85 == "gelu" else F.silu(pre85)
                            logits85 = F.linear(
                                hidden85,
                                R85["W"]["attr_heads.value_bin.2.weight"],
                                R85["W"]["attr_heads.value_bin.2.bias"]
                            ) + trans85
                            probs85 = logits85.softmax(-1)
                            pred85 = (probs85 * R85["centres"][ix85]).sum(-1)
                            for conditional85 in (False, True):
                                mean85 = pred85 / (1 - probs85[:, 0]).clamp_min(1e-8) if conditional85 else pred85
                                meanerr85 = float((
                                    (mean85 - R85["target_mean"][ix85]) /
                                    R85["scale"][ix85]
                                ).square().mean().sqrt())
                                best_nll85 = min(
                                    float((
                                        -probs85[torch.arange(len(ix85)), R85["bins"][os85][1][ix85]]
                                         .clamp_min(1e-30).log()
                                        - R85["target_nll"][ix85]
                                    ).square().mean().sqrt())
                                    for os85 in ("left", "right")
                                )
                                config85 = (
                                    flow_mode85, gap_mode85, side85, zp85, fp85,
                                    age_mode85, activation85, conditional85
                                )
                                results85.append((
                                    meanerr85 + 0.1 * best_nll85,
                                    meanerr85, best_nll85, config85
                                ))

results85.sort(key=lambda x: x[0])
R85["results"] = results85
print("=== TOP DETERMINISTIC CONFIGURATIONS ===")
for result85 in results85[:12]:
    print(result85)
print("Elapsed seconds:", round(time.time() - t85, 1))
print("Diagnostic edit remains:", edited())
```