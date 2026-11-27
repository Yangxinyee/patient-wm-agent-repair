The learned empty-action encoding resolves the mismatch: measurement posteriors now match within about \(3\times10^{-5}\) RMS. That validates the reconstructed posterior, not the drug-effect estimates. I’ll check complete bundles next; agreement there would let me calculate targeted pathway edits rather than scale weights blindly.

```python
import numpy as np
import pandas as pd
import torch

repair_units = {
    "glucose_finger": 0, "glucose_lab": 5,
    "map_arterial": 2, "map_cuff": 2,
    "insulin": 3, "norepinephrine": 4,
    "hospital_admission": 0, "icu_admission": 0,
}
repair_rate_units = {"insulin": 2, "norepinephrine": 3}
repair_ekinds = {
    "glucose_finger": 1, "glucose_lab": 1,
    "map_arterial": 1, "map_cuff": 1,
    "insulin": 2, "norepinephrine": 2,
    "hospital_admission": 3, "icu_admission": 4,
}

repair_bank = r45_bank
repair_bundle_keys = pd.MultiIndex.from_arrays([
    repair_bank["stay_id"],
    np.rint(repair_bank["hour"] * 60).astype(int)
])
repair_records = r45_records.copy()
repair_records["minute"] = np.rint(repair_records["hour"] * 60).astype(int)
repair_record_keys = pd.MultiIndex.from_frame(
    repair_records[["stay_id", "minute"]])
repair_records["bundle_index"] = repair_bundle_keys.get_indexer(repair_record_keys)
assert repair_records["bundle_index"].ge(0).all()

repair_n = len(repair_bank["hour"])
repair_pools = {
    "encode": torch.zeros(repair_n, 192),
    "encode_action": torch.zeros(repair_n, 192),
}
repair_counts = {
    "encode": np.zeros(repair_n, dtype=np.int64),
    "encode_action": np.zeros(repair_n, dtype=np.int64),
}

for event, frame in repair_records.groupby("event", sort=False):
    is_action = event in ["insulin", "norepinephrine"]
    prefix = "encode_action" if is_action else "encode"
    atoms = atom_embedding_repair(
        event,
        values=frame["value"].to_numpy(),
        rates=frame["rate"].to_numpy(),
        doses=frame["dose"].to_numpy(),
        unit=repair_units[event],
        rate_unit=repair_rate_units.get(event, 0),
        ekind=repair_ekinds[event],
        side="left")
    idx_t = torch.as_tensor(frame["bundle_index"].to_numpy(), dtype=torch.long)
    repair_pools[prefix].index_add_(0, idx_t, rp_mlp(atoms, prefix + ".mlp1"))
    np.add.at(repair_counts[prefix], frame["bundle_index"].to_numpy(), 1)

repair_bundle_encodings = {}
for prefix in ["encode", "encode_action"]:
    count = repair_counts[prefix]
    pooled = repair_pools[prefix] / torch.as_tensor(
        np.sqrt(np.maximum(count, 1)), dtype=torch.float32)[:, None]
    pooled += rp_linear(
        torch.as_tensor(np.log1p(count), dtype=torch.float32)[:, None],
        prefix + ".size")
    repair_bundle_encodings[prefix] = rp_norm(
        rp_mlp(pooled, prefix + ".mlp2"), prefix + ".norm")

repair_e = repair_bundle_encodings["encode"]
repair_u = repair_bundle_encodings["encode_action"]
repair_valid = np.flatnonzero(
    np.r_[False, repair_bank["stay_id"][1:] == repair_bank["stay_id"][:-1]])
repair_prev = torch.as_tensor(
    repair_bank["x"][repair_valid - 1], dtype=torch.float32)
repair_dt = torch.as_tensor(
    (np.rint(repair_bank["hour"][repair_valid] * 60)
     - np.rint(repair_bank["hour"][repair_valid - 1] * 60)) * 60,
    dtype=torch.float32)
repair_flowed_h = repair_prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -repair_dt[:, None, None]
    * torch.exp(rW["core.flow.a"])[None, None, :])
repair_ctx = (
    repair_prev[:, :192]
    + rp_norm(repair_flowed_h.mean(dim=1), "core.out_norm")
    + rp_mlp(torch.log1p(repair_dt)[:, None] / 10, "gap")
)
repair_z = rp_mlp(
    torch.cat([repair_ctx, repair_e[repair_valid], repair_u[repair_valid]], dim=1),
    "z_post")[:, :32]
repair_target_z = torch.as_tensor(
    repair_bank["x"][repair_valid, 960:992], dtype=torch.float32)
repair_z_error = ((repair_z - repair_target_z) ** 2).mean(dim=1).sqrt().numpy()

print("=== COMPLETE-BUNDLE POSTERIOR CHECK ===")
print("overall RMS:", float(np.sqrt(np.mean(repair_z_error ** 2))),
      "median:", float(np.median(repair_z_error)),
      "max:", float(repair_z_error.max()))
for label, mask in [
    ("measurement/visit only", repair_counts["encode_action"][repair_valid] == 0),
    ("action only", repair_counts["encode"][repair_valid] == 0),
    ("multiple atoms", sum(repair_counts.values())[repair_valid] > 1),
]:
    print(label, "n:", int(mask.sum()),
          "RMS:", float(np.sqrt(np.mean(repair_z_error[mask] ** 2))))
worst = np.argsort(repair_z_error)[-5:]
print(pd.DataFrame({
    "stay": repair_bank["stay_id"][repair_valid[worst]],
    "hour": repair_bank["hour"][repair_valid[worst]],
    "n_world": repair_counts["encode"][repair_valid[worst]],
    "n_action": repair_counts["encode_action"][repair_valid[worst]],
    "posterior_RMS": repair_z_error[worst],
}).to_string(index=False))

print("\n=== RECURRENT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("core.prelude.", "core.blocks.", "core.coda.", "z_in.")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 51 result / Run 52\n"
            "Learned empty action encoding (including size bias) resolves measurement "
            "posterior mismatch to ~2.6e-5 RMS. Measurement units: finger=0, lab=5, MAP=2; "
            "left bin convention. Constructed complete bundle encodings and checked all "
            "posteriors before recurrent reconstruction. No edits.\n")
```
