No tested unit convention achieved numerical agreement, so unit IDs alone do not explain the mismatch. I’ll check for stale weight caches from earlier probes and repeat the isolated-reading test with freshly loaded originals. Restored agreement would identify a cache problem; otherwise I’ll stop treating this reconstruction as validated.

```python
import torch
import numpy as np
import pandas as pd
import dis

print("=== CACHED-WEIGHT AUDIT ===")
differences79 = []
for name79 in NAMES:
    if name79 not in repair_weights:
        continue
    original79 = torch.as_tensor(load_original(name79), dtype=torch.float32)
    difference79 = float((repair_weights[name79] - original79).abs().max())
    if difference79 > 1e-7:
        differences79.append((name79, difference79))
print("non-original cached tensors:", differences79[:30])

repair_weights = {
    name79: torch.as_tensor(load_original(name79), dtype=torch.float32).clone()
    for name79 in NAMES
}
latent_empty_e = empty_set58("encode").reshape(192)

print("\n=== VOCABULARY CONVENTIONS ===")
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("config:", VOCAB["config"])
print("numeric note:", str(VOCAB["numeric_note"])[:1600])

print("\n=== SMALL HELPER IMPLEMENTATIONS ===")
for name79 in ("repair_linear", "repair_mlp_act", "empty_set58", "single_set57"):
    print("\n", name79)
    for ins79 in dis.get_instructions(globals()[name79]):
        print(ins79.offset, ins79.opname, ins79.argrepr[:90])

ctx79 = latent_context(indices78)
empty_u79 = empty_set58("encode_action").reshape(1, 192)
scores79 = []
for unit79 in range(6):
    atoms79 = torch.stack([
        complete_atom57(row79, kind=0, ekind=1, unit=unit79)
        for _, _, row79 in isolated78
    ])
    e79 = single_set57(atoms79, "encode")
    for absent79 in ("encoded", "zero"):
        u79 = (empty_u79.expand(len(indices78), -1)
               if absent79 == "encoded"
               else torch.zeros(len(indices78), 192))
        z79 = latent_post_mean(ctx79, e79, u79)
        rms79 = (z79 - truth78).square().mean(1).sqrt().numpy()
        for event79 in sorted(set(x[0] for x in isolated78)):
            mask79 = np.array([x[0] == event79 for x in isolated78])
            scores79.append({
                "event": event79, "unit": unit79, "absent_action": absent79,
                "mean_rms": float(rms79[mask79].mean()),
                "max_rms": float(rms79[mask79].max())
            })
unit_table79 = pd.DataFrame(scores79).sort_values(["event", "mean_rms"])
print("\n=== BEST CONVENTION WITH FRESH ORIGINAL WEIGHTS ===")
print(unit_table79.groupby("event", sort=False).head(1).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 79; forward 6 — original-cache audit\n"
        f"- Non-original cached tensors found: {differences79}.\n"
        "- Reloaded original weights into the local reconstruction cache and "
        "repeated isolated-reading convention tests. No submitted weights changed.\n"
    )
```