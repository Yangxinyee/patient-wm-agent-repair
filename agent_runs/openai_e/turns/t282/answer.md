The prelude reconstruction matches the cached features exactly, but the reconstructed posterior differs from the GPU posterior. I’ll anchor at the GPU state and apply only the dose-induced posterior change. Factor-one reconstruction and agreement with the earlier contrasts must hold before I trust a new fit.

```python
from pathlib import Path
import numpy as np, torch

def np243(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

sample243 = np.unique(np.r_[
    first_indices198[:3],
    drug_indices242[np.linspace(0, len(drug_indices242)-1, 25).astype(int)]
]).astype(int)

def encoded_bundles243(ids, drug=None, factor=1.0,
                       end_numeric="absent", end_units="none",
                       world_only=False, empty_observation=False):
    observation_lists, action_lists = [], []
    for i in ids:
        observations, actions = [], []
        key = (int(sid191[i]), int(minute191[i]))
        for row, kind in entries191[key]:
            if row.event == drug:
                row = row._replace(rate=row.rate*factor, dose=row.dose*factor)
            atom = recon_record_atom(
                row, kind=kind, end_numeric=end_numeric, end_units=end_units
            )
            is_action = (
                row.event in ("insulin", "norepinephrine")
                and (not world_only or kind == 0)
            )
            (actions if is_action else observations).append(atom)
        observation_lists.append(observations)
        action_lists.append(actions)
    return (
        recon_sets(observation_lists, prefix="encode",
                   empty_zero=empty_observation),
        recon_sets(action_lists, prefix="encode_action", empty_zero=True)
    )

rules243 = []
for end_numeric, end_units in (
    ("absent", "none"), ("absent", "all"),
    ("zero", "none"), ("all", "all")
):
    for world_only in (False, True):
        for empty_observation in (False, True):
            ee, uu = encoded_bundles243(
                sample243, end_numeric=end_numeric, end_units=end_units,
                world_only=world_only, empty_observation=empty_observation
            )
            error = float(np.mean((np243(ee)-np243(obs191)[sample243])**2)
                          + np.mean((np243(uu)-np243(act191)[sample243])**2))
            rules243.append((error, end_numeric, end_units,
                             world_only, empty_observation))
rules243.sort(key=lambda item: item[0])
print("BEST ENCODING RULES")
for rule in rules243[:5]:
    print(rule)

_, en243, eu243, wo243, eo243 = rules243[0]
assert rules243[0][0] < 1e-10, "Factual encoding is not aligned."
rule243 = dict(end_numeric=en243, end_units=eu243,
               world_only=wo243, empty_observation=eo243)

key243 = ("norepinephrine", 2.0)
ids243 = np.asarray(cf_train192[key243]["ids"], dtype=int)
ef243, uf243 = encoded_bundles243(ids243, **rule243)
ec243, uc243 = encoded_bundles243(
    ids243, drug=key243[0], factor=key243[1], **rule243
)
print("\nALL NE FACTOR-ONE ENCODING ERRORS",
      float(np.max(np.abs(np243(ef243)-np243(obs191)[ids243]))),
      float(np.max(np.abs(np243(uf243)-np243(act191)[ids243]))))

zf243 = posterior94(ctx_base191[ids243], ef243, uf243)
zc243 = z_base191[ids243] + (
    posterior94(ctx_base191[ids243], ec243, uc243) - zf243
)
hidden_ne2_243, raw_ne2_243 = prelude_hidden93(
    h_base191[ids243], ec243, uc243, zc243
)
E_ne2_243 = np243(core_norm(
    torch.from_numpy(raw_ne2_243), "core.prelude_norm"
))
reference243 = np243(cf_train192[key243]["E"])
print("NE2 PRELUDE AGREEMENT",
      "RMS", float(np.sqrt(np.mean((E_ne2_243-reference243)**2))),
      "max", float(np.max(np.abs(E_ne2_243-reference243))))
print("NE2 HIDDEN CONTRAST RMS",
      float(np.sqrt(np.mean(
          (hidden_ne2_243-hidden_base242[ids243])**2
      ))))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 243: GPU-anchored dose-feature reconstruction\n"
        "Run242 normalized prelude reconstruction was exact, but absolute "
        "posterior reconstruction errors were material (baseline all RMS "
        ".3167; drug RMS .04390). Testing exact factual atom/set conventions "
        "and anchoring synthetic posterior changes at GPU factual z. "
        "No weights changed; joint236 retained.\n"
        "Python runs: 243; forwards: 29.\n"
    )
```