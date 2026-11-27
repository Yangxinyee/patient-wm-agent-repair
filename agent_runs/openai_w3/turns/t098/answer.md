The prelude replay returns the expected features. The action helper, however, returns only selected bundles, so treating its output as the full stream caused an indexing error. I’ll rebuild the actions with explicit bundle ownership and exclude infusion-end markers. Agreement with the factual action vectors would confirm the mapping; disagreement would block the repair fit.

```python
import numpy as np
import torch

print("BUNDLE STRUCTURE", len(r40_info), repr(r40_info[int(idx81[0])])[:500])
print("SELECTED ACTION INDICES", np.asarray(r41_action_idx).shape,
      np.asarray(r41_action_idx)[:8])

def action_vectors87(factors=(1.0, 1.0)):
    names87 = ("insulin", "norepinephrine")
    atoms87, owners87 = [], []
    counts87 = np.zeros(len(idx81), dtype=np.float32)
    for local87, global87 in enumerate(idx81):
        for row87, is_end87 in r40_info[int(global87)]:
            if is_end87 or row87.event not in names87:
                continue
            mult87 = float(factors[names87.index(row87.event)])
            changed87 = row87._replace(rate=float(row87.rate)*mult87,
                                      dose=float(row87.dose)*mult87)
            atoms87.append(atom80(changed87, False, "clear_all"))
            owners87.append(local87)
            counts87[local87] += 1
    summed87 = torch.zeros((len(idx81), 192), dtype=torch.float32)
    if atoms87:
        hidden87 = mlp(torch.stack(atoms87), "encode_action.mlp1")
        summed87.index_add_(0, torch.tensor(owners87, dtype=torch.long), hidden87)
    count87 = torch.from_numpy(counts87)[:, None]
    return layer_norm(
        mlp(summed87 / count87.clamp_min(1).sqrt()
            + linear(torch.log1p(count87), "encode_action.size"),
            "encode_action.mlp2"),
        "encode_action.norm"
    )

u87 = action_vectors87()
print("FACTUAL ACTION RMSE", float(torch.mean((u87-u81)**2).sqrt()))
print("DRUG ACTION RMSE", float(torch.mean((u87[active81]-u81[active81])**2).sqrt()))
print("ACTIVE BUNDLES", int(np.sum(active81)))
print("RATE LABEL QUANTILES", np.quantile(rates81[idx81], [0, .5, .9, .99, 1], axis=0))

print("DIRECTION/FIT ARRAY NAMES")
for k in sorted(list(globals())):
    v = globals()[k]
    if any(s in k.lower() for s in ("dir", "coef", "slope")) and isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 86-87; forwards 7. Run 86 found fn76 returns selected action bundles, not the full stream; its indexing diagnostic failed without changing weights. Built an explicitly indexed WORLD-drug action re-encoder, excluding END markers, to validate prelude dose contrasts. No edits.\n")
```