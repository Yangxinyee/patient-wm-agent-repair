The identification checks support an immediate insulin effect near −1.95 and norepinephrine near +3.1, with different delayed responses. I’ll now validate every drug bundle, including infusion ends. A near-zero posterior reconstruction error across all bundles would let us test weight corrections without encoding uncertainty.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

# Use the float32 boundary convention established in run 45.
for spec49 in num42.values():
    spec49["fine_bin_edges"] = np.asarray(
        spec49["fine_bin_edges"], dtype=np.float32
    )

def atom_embedding49(drugs, rates, doses, kind=0, ekind=1,
                     units="native", attributes="full", action_category=0):
    n = len(drugs)
    numbers = np.zeros((n, 6), dtype=np.float32)
    bin_ids = {a: np.zeros(n, dtype=np.int64)
               for a in ("value", "dose", "rate")}
    if attributes != "missing":
        for j, attr in ((2, "dose"), (4, "rate")):
            vals = np.asarray(doses if attr == "dose" else rates,
                              dtype=np.float32)
            if attributes == "zero":
                vals = np.zeros_like(vals)
            for drug in ("insulin", "norepinephrine"):
                mask = np.asarray(drugs) == drug
                spec = num42[(drug, attr)]
                numbers[mask, j] = np.clip(
                    (vals[mask].astype(np.float64) - spec["median"])
                    / spec["iqr"], -10, 10
                )
                numbers[mask, j + 1] = 1
                bin_ids[attr][mask] = (
                    np.searchsorted(spec["fine_bin_edges"],
                                    vals[mask], side="left") + 1
                )
    code_ids = torch.tensor([code42[d] for d in drugs], dtype=torch.long)
    emb = (wt42("embed.code.weight")[code_ids]
           + wt42("embed.kind.weight")[kind]
           + wt42("embed.ekind.weight")[ekind]
           + defaults42)
    if units == "native":
        unit_ids = torch.tensor(
            [3 if d == "insulin" else 4 for d in drugs], dtype=torch.long)
        rate_ids = torch.tensor(
            [2 if d == "insulin" else 3 for d in drugs], dtype=torch.long)
    else:
        unit_ids = rate_ids = torch.zeros(n, dtype=torch.long)
    emb = (emb
           + wt42("embed.categorical.unit.weight")[unit_ids]
           + wt42("embed.categorical.rate_unit.weight")[rate_ids]
           + wt42("embed.kq.weight")[0])
    if action_category:
        emb = (emb
               - wt42("embed.categorical.action.weight")[0]
               + wt42("embed.categorical.action.weight")[action_category])
    for attr in ("value", "dose", "rate"):
        emb = emb + wt42(f"embed.bins.{attr}.weight")[
            torch.tensor(bin_ids[attr], dtype=torch.long)]
    return emb + F.linear(torch.tensor(numbers),
                          wt42("embed.numeric.weight"),
                          wt42("embed.numeric.bias"))

def pool_actions49(emb, owners, counts):
    transformed = net42(emb, "encode_action.mlp1", "gelu")
    pooled = torch.zeros((len(counts), 192), dtype=torch.float32)
    pooled.index_add_(0, torch.tensor(owners, dtype=torch.long), transformed)
    c = torch.as_tensor(counts, dtype=torch.float32)
    pooled = (pooled / torch.sqrt(c[:, None])
              + affine42(torch.log1p(c[:, None]), "encode_action.size"))
    return ln42(net42(pooled, "encode_action.mlp2", "gelu"),
                "encode_action.norm")

# First require the generic implementation to match the validated start encoder.
ref_u49 = start_u43(layout44, umap44, kq=0)
start_meta49 = np.asarray(atommeta42, dtype=object)
start_drugs49 = start_meta49[:, 1].tolist()
start_rates49 = start_meta49[:, 2].astype(np.float32)
start_doses49 = start_meta49[:, 3].astype(np.float32)
start_owners49 = start_meta49[:, 0].astype(np.int64)
start_tests49 = []
for ek49 in range(wt42("embed.ekind.weight").shape[0]):
    emb49 = atom_embedding49(start_drugs49, start_rates49, start_doses49,
                             ekind=ek49)
    u49 = pool_actions49(emb49, start_owners49, counts42)
    start_tests49.append(
        (float(torch.sqrt(torch.mean((u49 - ref_u49) ** 2))), ek49)
    )
print("Start encoder agreement:", sorted(start_tests49))
start_ekind49 = min(start_tests49)[1]
assert min(start_tests49)[0] < 1e-4, "Generic start encoder needs correction"

s49 = dict(np.load("/work/fwd_4/states_original.npz"))
drug_indices49 = np.flatnonzero(
    s49["has_insulin"] | s49["has_norepinephrine"]
)
state_keys49 = {
    (int(s49["stay_id"][i]), int(round(float(s49["hour"][i]) * 10000))): j
    for j, i in enumerate(drug_indices49)
}
drug_records49 = train[
    train.stay_id.isin(np.unique(s49["stay_id"]))
    & train.event.isin(["insulin", "norepinephrine"])
].copy()

start_rows49, end_rows49 = [], []
for row49 in drug_records49.itertuples(index=False):
    start_key49 = (int(row49.stay_id), int(round(row49.hour * 10000)))
    end_key49 = (int(row49.stay_id), int(round(row49.end_hour * 10000)))
    atom49 = (row49.event, float(row49.rate), float(row49.dose))
    if start_key49 in state_keys49:
        start_rows49.append((state_keys49[start_key49], *atom49))
    if end_key49 in state_keys49:
        end_rows49.append((state_keys49[end_key49], *atom49))

def unpack49(rows):
    a = np.asarray(rows, dtype=object)
    return (a[:, 0].astype(np.int64), a[:, 1].tolist(),
            a[:, 2].astype(np.float32), a[:, 3].astype(np.float32))

so49, sd49, sr49, sx49 = unpack49(start_rows49)
eo49, ed49, er49, ex49 = unpack49(end_rows49)
counts_all49 = np.bincount(
    np.concatenate([so49, eo49]), minlength=len(drug_indices49)
)
print("Bundle counts match:",
      np.array_equal(counts_all49, s49["n_atoms"][drug_indices49]),
      "bundles", len(drug_indices49),
      "starts", len(so49), "ends", len(eo49))
assert np.array_equal(counts_all49, s49["n_atoms"][drug_indices49])

ctx_all49 = contexts37(s49, drug_indices49)
target_z49 = torch.tensor(s49["x"][drug_indices49, 960:],
                          dtype=torch.float32)
start_emb49 = atom_embedding49(sd49, sr49, sx49, ekind=start_ekind49)

# Choose the empty measurement vector by direct posterior agreement.
e_candidates49 = {"encoded_empty": emap43["encoded_empty"],
                  "zero": torch.zeros(192)}
for key49, v49 in globals().get("eoptions44", {}).items():
    if isinstance(v49, torch.Tensor) and v49.shape == (192,):
        e_candidates49[key49] = v49
e_tests49 = []
for key49, v49 in e_candidates49.items():
    inp49 = torch.cat([ctx42, v49.expand(len(ctx42), -1), ref_u49], dim=1)
    z49 = net42(inp49, "z_post", "gelu")[:, :32]
    e_tests49.append(
        (float(torch.sqrt(torch.mean((z49 - targetz42) ** 2))), key49)
    )
empty_key49 = min(e_tests49)[1]
empty_e49 = e_candidates49[empty_key49]
print("Empty measurement agreement:", sorted(e_tests49))
base_post49 = F.linear(
    torch.cat([ctx_all49, empty_e49.expand(len(ctx_all49), -1)], dim=1),
    wt42("z_post.0.weight")[:, :384], wt42("z_post.0.bias")
)

end_results49 = []
n_action_categories49 = wt42("embed.categorical.action.weight").shape[0]
for attrs49 in ("missing", "full", "zero"):
    for units49 in ("zero", "native"):
        for ek49 in range(wt42("embed.ekind.weight").shape[0]):
            for ac49 in range(n_action_categories49):
                end_emb49 = atom_embedding49(
                    ed49, er49, ex49, kind=2, ekind=ek49, units=units49,
                    attributes=attrs49, action_category=ac49
                )
                u49 = pool_actions49(
                    torch.cat([start_emb49, end_emb49]),
                    np.concatenate([so49, eo49]), counts_all49
                )
                h49 = F.gelu(base_post49 + F.linear(
                    u49, wt42("z_post.0.weight")[:, 384:], None))
                z49 = affine42(h49, "z_post.2")[:, :32]
                rms49 = float(torch.sqrt(torch.mean((z49 - target_z49) ** 2)))
                end_results49.append(
                    (rms49, (attrs49, units49, ek49, ac49))
                )

print("Best END conventions:", sorted(end_results49)[:8])
best_end49 = min(end_results49)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 49 / all-action reconstruction\n"
        f"Generic start encoder test: {min(start_tests49)}. "
        f"Best END convention: {best_end49}. "
        f"Validated bundle counts for {len(drug_indices49)} drug bundles. "
        "No tensors edited.\n"
    )
```