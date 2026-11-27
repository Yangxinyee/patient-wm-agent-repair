The posterior’s final layer has 192 input features, which is suitable for a small regression-based edit. I’ll first check reconstruction on both infusion starts and endings. Matching both to numerical precision would confirm that subsequent dose tests use the same inputs as the forward tool.

```python
import torch
import itertools
from collections import defaultdict

torch.set_num_threads(8)
torch.set_grad_enabled(False)

latent_state = np.asarray(state_archive40["x"], dtype=np.float32)
latent_sid = np.asarray(state_archive40["stay_id"])
latent_hour = np.asarray(state_archive40["hour"])
latent_stays = np.unique(latent_sid)
latent_records = train[train.stay_id.isin(latent_stays)].copy()

latent_starts = defaultdict(list)
latent_ends = defaultdict(list)
for rr41 in latent_records[latent_records.event.isin(DRUGS)].to_dict("records"):
    latent_starts[(int(rr41["stay_id"]), round(float(rr41["hour"]), 4))].append(rr41)
    latent_ends[(int(rr41["stay_id"]), round(float(rr41["end_hour"]), 4))].append(rr41)

latent_bundles = []
for ii41, (ss41, hh41) in enumerate(zip(latent_sid, latent_hour)):
    kk41 = (int(ss41), round(float(hh41), 4))
    starts41, ends41 = latent_starts[kk41], latent_ends[kk41]
    if (starts41 or ends41) and ii41 > 0 and latent_sid[ii41-1] == ss41:
        latent_bundles.append((ii41, starts41, ends41))
print("Action bundles:", len(latent_bundles))
print("Start/end counts:", pd.Series(
    [(len(s), len(e)) for _, s, e in latent_bundles]).value_counts().to_dict())

gap_prefix41 = next(n[:-len(".0.weight")] for n in NAMES
                    if "gap" in n and n.endswith(".0.weight"))
print("Gap prefix:", gap_prefix41, "empty encodings:", list(repair_empty_e))
empty_key41 = "empty_gelu" if "empty_gelu" in repair_empty_e else "gelu"
latent_empty_e = repair_empty_e[empty_key41]

def latent_context(indices):
    indices = np.asarray(indices, dtype=np.int64)
    prev = torch.from_numpy(latent_state[indices-1, :960])
    dt = torch.from_numpy(
        ((latent_hour[indices] - latent_hour[indices-1]) * 3600).astype(np.float32))
    hf = prev[:, 192:].reshape(-1, 4, 192) * torch.exp(
        -dt[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :])
    return (prev[:, :192] + repair_norm(hf.mean(1), "core.out_norm")
            + repair_mlp_act(torch.log(dt[:, None]) / 10, gap_prefix41, F.gelu))

def latent_categories(unit, rate_unit, action=0, status=0, route=0, discharged_to=0):
    chosen = dict(unit=unit, rate_unit=rate_unit, action=action, status=status,
                  route=route, discharged_to=discharged_to)
    return sum(repair_weights[f"embed.categorical.{k}.weight"][v]
               for k, v in chosen.items())

def latent_action_encode(atom_vectors):
    aa = torch.stack(atom_vectors)
    pooled = repair_mlp_act(aa, "encode_action.mlp1", F.gelu).sum(0) / np.sqrt(len(aa))
    pooled = pooled + repair_linear(
        torch.tensor([np.log1p(len(aa))], dtype=torch.float32), "encode_action.size")
    return repair_norm(repair_mlp_act(pooled, "encode_action.mlp2", F.gelu),
                       "encode_action.norm")

def latent_hidden(context, action):
    ee = latent_empty_e.expand(context.shape[0], -1)
    return F.gelu(repair_linear(torch.cat([context, ee, action], -1), "z_post.0"))

single_start41 = [(i, s[0]) for i, s, e in latent_bundles if len(s) == 1 and not e][:80]
idx41 = np.array([i for i, _ in single_start41])
u41 = torch.stack([
    latent_action_encode([repair_atom_base(r) + latent_categories(
        3 if r["event"] == "insulin" else 4,
        2 if r["event"] == "insulin" else 3)])
    for _, r in single_start41
])
pred41 = repair_linear(latent_hidden(latent_context(idx41), u41), "z_post.2")[:, :32]
print("Start reconstruction RMS:",
      float(torch.mean((pred41 - torch.from_numpy(latent_state[idx41, 960:]))**2).sqrt()))

single_end41 = [(i, e[0]) for i, s, e in latent_bundles if not s and len(e) == 1][:64]
print("Single END examples:", len(single_end41))
end_search41 = []
if single_end41:
    end_idx41 = np.array([i for i, _ in single_end41])
    end_ctx41 = latent_context(end_idx41)
    end_target41 = torch.from_numpy(latent_state[end_idx41, 960:])
    for numbers41, unit41, rate_unit41 in itertools.product(
            ["absent", "copied", "zero"], range(6), range(4)):
        atoms41 = []
        for _, source41 in single_end41:
            row41 = dict(source41)
            if numbers41 == "zero":
                row41["rate"] = row41["dose"] = 0.0
            atoms41.append(repair_atom_base(
                row41, kind=2, include_numbers=numbers41 != "absent")
                + latent_categories(unit41, rate_unit41))
        atom_hidden41 = repair_mlp_act(torch.stack(atoms41), "encode_action.mlp1", F.gelu)
        pool41 = atom_hidden41 + repair_linear(
            torch.full((len(atoms41), 1), np.log(2), dtype=torch.float32),
            "encode_action.size")
        uu41 = repair_norm(repair_mlp_act(pool41, "encode_action.mlp2", F.gelu),
                           "encode_action.norm")
        zz41 = repair_linear(latent_hidden(end_ctx41, uu41), "z_post.2")[:, :32]
        rms41 = float(torch.mean((zz41-end_target41)**2).sqrt())
        end_search41.append((rms41, numbers41, unit41, rate_unit41))
    print("Best END conventions:", sorted(end_search41)[:8])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 41; forward 6 — full action encoding check\n"
            f"- Action bundles: {len(latent_bundles)}; single-END examples: {len(single_end41)}.\n"
            f"- Best END conventions: {sorted(end_search41)[:3]}.\n"
            "- No weights changed.\n")
```