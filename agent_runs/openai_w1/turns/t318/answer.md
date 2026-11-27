The independent prelude matches the replay helper exactly. The old projections produce very large signals on the corrected factual features, so they should not be reused. I’ll rebuild dose-edited action vectors and obtain states from the currently installed model. Small factual projection errors on those states—not merely on original-model states—will be required before testing another edit.

```python
import numpy as np, torch
from pathlib import Path

actidx279 = np.flatnonzero(np276(rr_act_mask112))
atoms279 = torch.as_tensor(rr_atoms4, dtype=torch.float32).clone()
for i in actidx279:
    atoms279[int(i)] = torch.as_tensor(rr_atom(**rr_desc4[int(i)]), dtype=torch.float32)
ubase279 = rr_encode_atoms112(atoms279, rr_act_mask112, "encode_action", True).detach()
print("rebuilt baseline action versus anchored action RMS:",
      float(torch.sqrt(torch.mean((ubase279-u277)**2))))
print("action atoms:", len(actidx279),
      "END atoms:", sum(bool(rr_desc4[int(i)]["end"]) for i in actidx279))

Ucf279 = {}
for drug279 in ["insulin", "norepinephrine"]:
    di279 = [int(i) for i in actidx279 if rr_desc4[int(i)]["code"] == drug279]
    for factor279 in [0.0, 0.5, 1.5, 2.0, 3.0]:
        aa279 = atoms279.clone()
        for i in di279:
            d279 = rr_desc4[i].copy()
            d279["rate"] *= factor279
            d279["dose"] *= factor279
            aa279[i] = torch.as_tensor(rr_atom(**d279), dtype=torch.float32)
        uu279 = rr_encode_atoms112(aa279, rr_act_mask112, "encode_action", True).detach()
        Ucf279[(drug279, factor279)] = uu279 + (u277-ubase279)
        print(drug279, factor279, "action contrast RMS",
              float(torch.sqrt(torch.mean((uu279-ubase279)**2))))

ids279 = sorted(int(x) for x in np.unique(rr_state4["stay_id"]))
print("\n=== EXACT TRAINING STAYS FOR GPU STATE EXPORT ===")
print(ids279)
print("count:", len(ids279))
print("ds_rates alignment:",
      bool(np.array_equal(np276(ds_s["stay_id"]), np276(rr_state4["stay_id"]))),
      bool(np.allclose(np276(ds_s["hour"]), np276(rr_state4["hour"]))))
print("physical-rate ranges:", np.nanmin(np276(ds_rates), axis=0),
      np.nanmax(np276(ds_rates), axis=0))
print("current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 279\nIndependent prelude matches replay exactly. Corrected E exposes old projection leakage RMS[46.51,9.36], so old dose projections are rejected. Rebuilt dose-edited action vectors for x0/.5/1.5/2/3, anchored to factual action encoding. Preparing GPU states for current safe236 on the same 100 training stays, to fit on actual edited-model trajectories. Counts: 279 Python runs, 29 forwards.\n")
```