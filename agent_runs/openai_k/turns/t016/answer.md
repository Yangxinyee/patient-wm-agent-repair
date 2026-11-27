The small-sample factual errors suggest missing drug responses, but their uncertainty is too large to set an edit’s magnitude. Pre/post-state contrasts also recover little dose innovation. I’ll now reconstruct the action encoder and posterior calculation from the supplied tensors. Agreement with the saved posterior states would make analytical upstream edits testable; disagreement would mean the reconstruction is not yet trustworthy.

```python
import torch.nn.functional as F

torch.set_num_threads(8)
torch.set_grad_enabled(False)
W0 = {nm: load_original(nm).float().cpu() for nm in NAMES}

def lin0(x, prefix):
    return F.linear(x, W0[prefix + ".weight"], W0.get(prefix + ".bias"))

def act0(x, kind="gelu"):
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    return F.relu(x)

def mlp0(x, prefix, kind="gelu"):
    return lin0(act0(lin0(x, prefix + ".0"), kind), prefix + ".2")

def norm0(x, prefix):
    return F.layer_norm(x, (192,), W0[prefix + ".weight"],
                        W0[prefix + ".bias"], 1e-5)

code_id0 = {v["name"]: v["id"] for v in VOCAB["codes"]}
num_spec0 = {(v["code"], v["attribute"]): v for v in VOCAB["numeric"]}

print("=== EMBEDDING ROW NORMS ===")
for nm in NAMES:
    if nm.startswith(("embed.categorical.", "embed.kq.", "embed.text.",
                      "embed.kind.", "embed.ekind.", "embed.bins.")):
        print(nm, np.round(W0[nm].norm(dim=1).numpy(), 3))
print("Infusion unit strings:",
      df[df.event.isin(["insulin", "norepinephrine"])][
          ["event", "rate_unit", "dose_unit"]].drop_duplicates().to_dict("records"))

first_drug0 = []
for sid in np.unique(train_states0["stay_id"]):
    r = np.flatnonzero(
        (train_states0["stay_id"] == sid) &
        (train_states0["has_insulin"] | train_states0["has_norepinephrine"])
    )
    if len(r) and train_states0["n_atoms"][r[0]] == 1:
        first_drug0.append(r[0])
first_drug0 = np.asarray(first_drug0, dtype=int)
print("Single-atom first drug bundles:", len(first_drug0))

posterior_reconstruction0 = {}
posterior_candidates0 = {}
for drug in ("insulin", "norepinephrine"):
    r = first_drug0[train_states0["has_" + drug][first_drug0]]
    if len(r) == 0:
        print(drug, "no eligible bundles")
        continue
    pp = P.index.get_indexer(pd.MultiIndex.from_arrays([
        train_states0["stay_id"][r],
        np.floor(train_states0["hour"][r]).astype(int)
    ]))
    rr = rates[drug].to_numpy()[pp]
    assert np.all(rr > 0)
    dd = rr * (1.0 if drug == "insulin" else 4.8)
    nums = np.zeros((len(r), 6), dtype=np.float32)
    bins = {}
    for attr, vv, zi, hi in (("dose", dd, 2, 3), ("rate", rr, 4, 5)):
        sp = num_spec0[(drug, attr)]
        nums[:, zi] = np.clip((vv - sp["median"]) / sp["iqr"], -10, 10)
        nums[:, hi] = 1
        bins[attr] = torch.tensor(
            1 + np.searchsorted(sp["fine_bin_edges"], vv, side="left"),
            dtype=torch.long
        )
    base = (
        W0["embed.code.weight"][code_id0[drug]] +
        W0["embed.kind.weight"][0] +
        W0["embed.ekind.weight"][2] +
        W0["embed.bins.value.weight"][0] +
        W0["embed.bins.dose.weight"][bins["dose"]] +
        W0["embed.bins.rate.weight"][bins["rate"]] +
        lin0(torch.tensor(nums), "embed.numeric")
    )
    extras = torch.zeros(192)
    for nm in NAMES:
        if nm.startswith("embed.categorical.") and nm not in (
            "embed.categorical.unit.weight", "embed.categorical.rate_unit.weight"
        ):
            extras += W0[nm][0]
    extras += W0["embed.kq.weight"][0] + W0["embed.text.weight"][0]
    dt = torch.tensor(
        (train_states0["hour"][r] - train_states0["hour"][r - 1]) * 3600,
        dtype=torch.float32
    )
    previous = torch.tensor(train_states0["x"][r - 1])
    flowed_h = previous[:, 192:960].reshape(-1, 4, 192) * torch.exp(
        -dt[:, None, None] * W0["core.flow.a"].exp()[None, None, :]
    )
    target = torch.tensor(train_states0["x"][r, 960:992])
    candidates = []
    for activation in ("gelu", "silu", "relu"):
        for gap_rule in ("log", "log1p"):
            gx = dt.clamp_min(1).log() if gap_rule == "log" else dt.log1p()
            ctx = previous[:, :192] + norm0(flowed_h.mean(1), "core.out_norm") + \
                  mlp0(gx[:, None] / 10, "gap", activation)
            for include_defaults in (False, True):
                for unit in range(6):
                    for rate_unit in range(4):
                        atom = base + (
                            extras if include_defaults else torch.zeros_like(extras)
                        ) + W0["embed.categorical.unit.weight"][unit] + \
                            W0["embed.categorical.rate_unit.weight"][rate_unit]
                        summed = mlp0(atom, "encode_action.mlp1", activation) + \
                                 lin0(torch.full((len(r), 1), np.log(2),
                                                 dtype=torch.float32),
                                      "encode_action.size")
                        u = norm0(mlp0(summed, "encode_action.mlp2", activation),
                                  "encode_action.norm")
                        raw = mlp0(torch.cat([ctx, torch.zeros_like(u), u], 1),
                                   "z_post", activation)
                        for layout, mean in (
                            ("chunk", raw[:, :32]),
                            ("interleaved", raw[:, 0::2])
                        ):
                            error = float(((mean - target) ** 2).mean().sqrt())
                            candidates.append((error, activation, gap_rule,
                                               include_defaults, unit, rate_unit, layout))
    candidates.sort()
    posterior_candidates0[drug] = candidates
    posterior_reconstruction0[drug] = candidates[0]
    print("\n", drug, "target RMS:", round(float(target.square().mean().sqrt()), 5))
    print("Best reconstruction candidates (RMS, activation, gap, defaults, unit, rate unit, layout):")
    for row in candidates[:5]:
        print(row)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 11: action/posterior reconstruction\n"
        "- Small train-100 factual residuals suggest missing responses, but are not precise effect estimates.\n"
        "- Reconstructed stateless action and posterior primitives; compared directly to saved z means.\n"
        "- Best reconstruction configurations: " + repr(posterior_reconstruction0) + "\n"
        "- No weights edited.\n"
        "- Python runs: 11; forward passes: 4.\n"
    )
```