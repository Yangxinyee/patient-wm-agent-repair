The start encoder now matches exactly, but treating infusion ends as actions does not reproduce the posterior. That blocks a justified edit. I’ll test whether ends enter the observation encoder instead; an all-bundle error near numerical precision would confirm the correct split.

```python
import numpy as np
import torch
import torch.nn.functional as F

def encode_set50(emb, owners, counts, prefix, zero_if_empty=False):
    c = torch.as_tensor(counts, dtype=torch.float32)
    pooled = torch.zeros((len(c), 192), dtype=torch.float32)
    if len(owners):
        transformed = net42(emb, prefix + ".mlp1", "gelu")
        pooled.index_add_(0, torch.tensor(owners, dtype=torch.long), transformed)
    pooled = (pooled / torch.sqrt(c.clamp_min(1)[:, None])
              + affine42(torch.log1p(c[:, None]), prefix + ".size"))
    result = ln42(net42(pooled, prefix + ".mlp2", "gelu"), prefix + ".norm")
    if zero_if_empty:
        result[c == 0] = 0
    return result

start_counts50 = np.bincount(so49, minlength=len(drug_indices49))
end_counts50 = np.bincount(eo49, minlength=len(drug_indices49))
u_starts50 = encode_set50(start_emb49, so49, start_counts50,
                         "encode_action", zero_if_empty=True)
e_empty50 = encode_set50(
    torch.empty((0, 192)), np.empty(0, dtype=np.int64),
    np.zeros(len(drug_indices49)), "encode"
)
print("Empty encoder agreement:",
      float(torch.max(torch.abs(e_empty50 - empty_e49))))

def posterior50(e, u):
    return net42(torch.cat([ctx_all49, e, u], dim=1),
                 "z_post", "gelu")[:, :32]

z_ignore50 = posterior50(e_empty50, u_starts50)
err_ignore50 = z_ignore50 - target_z49
no_end50 = end_counts50 == 0
print("Ignore END, all-bundle RMS:",
      float(torch.sqrt(torch.mean(err_ignore50 ** 2))))
print("No-END subset RMS:",
      float(torch.sqrt(torch.mean(err_ignore50[no_end50] ** 2))),
      "n", int(no_end50.sum()))

partition_results50 = []
best_state50 = None
for attrs50 in ("missing", "zero", "full"):
    for units50 in ("zero", "native"):
        for ek50 in range(wt42("embed.ekind.weight").shape[0]):
            for ac50 in range(
                    wt42("embed.categorical.action.weight").shape[0]):
                end_emb50 = atom_embedding49(
                    ed49, er49, ex49, kind=2, ekind=ek50,
                    units=units50, attributes=attrs50,
                    action_category=ac50
                )
                e50 = encode_set50(end_emb50, eo49, end_counts50, "encode")
                z50 = posterior50(e50, u_starts50)
                rms50 = float(torch.sqrt(torch.mean((z50 - target_z49) ** 2)))
                cfg50 = (attrs50, units50, ek50, ac50)
                partition_results50.append((rms50, cfg50))
                if best_state50 is None or rms50 < best_state50[0]:
                    best_state50 = (rms50, cfg50, e50.clone(), z50.clone())

print("Best END-as-observation conventions:",
      sorted(partition_results50)[:10])
best_partition50 = min(partition_results50)
for label50, mask50 in (
    ("no END", no_end50),
    ("END and start", (end_counts50 > 0) & (start_counts50 > 0)),
    ("END only", (end_counts50 > 0) & (start_counts50 == 0))
):
    if np.any(mask50):
        error50 = best_state50[3][mask50] - target_z49[mask50]
        print(label50, int(mask50.sum()),
              "RMS", float(torch.sqrt(torch.mean(error50 ** 2))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 50 / END partition test\n"
        "Generic WORLD start encoder matches validated encoder to 1e-7; "
        "start event-kind index is 2. Treating all END atoms as actions failed "
        f"(best RMS {best_end49[0]:.6g}). "
        f"END-as-observation best: {best_partition50}. "
        "No weight edit until all-bundle reconstruction is validated.\n"
    )
```