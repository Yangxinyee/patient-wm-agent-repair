The whole-trajectory reconstruction stopped at an atom-count mismatch, before any fitting or editing. I’ll audit the missing atoms, including visit end markers, and continue only if the reconstructed bundle counts match exactly. Otherwise, the input construction still needs correction.

```python
import numpy as np
import torch
import torch.nn.functional as F

mismatch52 = np.flatnonzero(
    obs_counts51 + action_counts51 != s51["n_atoms"]
)
print("Before repair: mismatched bundles", len(mismatch52))
for i52 in mismatch52[:10]:
    print(int(s51["stay_id"][i52]), float(s51["hour"][i52]),
          "exported", int(s51["n_atoms"][i52]),
          "reconstructed", int(obs_counts51[i52] + action_counts51[i52]))
print("Non-drug end markers:")
print(world51[world51.end_hour.notna()].head(10)[
    ["stay_id", "hour", "event", "end_hour"]].to_string(index=False))

# Non-drug visit records can also carry end markers.
added52 = 0
for row52 in world51.itertuples(index=False):
    if np.isfinite(row52.end_hour):
        key52 = (int(row52.stay_id), int(round(row52.end_hour * 10000)))
        if key52 in all_keys51:
            obs_meta51.append(
                (all_keys51[key52], row52.event, np.nan, 2))
            added52 += 1

obs_array51 = np.asarray(obs_meta51, dtype=object)
obs_owner51 = obs_array51[:, 0].astype(np.int64)
obs_codes51 = obs_array51[:, 1].astype(str)
obs_values51 = obs_array51[:, 2].astype(np.float32)
obs_kinds51 = obs_array51[:, 3].astype(np.int64)
obs_counts51 = np.bincount(obs_owner51, minlength=len(s51["hour"]))
remaining52 = np.flatnonzero(
    obs_counts51 + action_counts51 != s51["n_atoms"])
print("Added markers:", added52, "remaining mismatches:", len(remaining52))
for i52 in remaining52[:8]:
    print(int(s51["stay_id"][i52]), float(s51["hour"][i52]),
          int(s51["n_atoms"][i52]),
          int(obs_counts51[i52] + action_counts51[i52]))
assert len(remaining52) == 0, "Bundle accounting is not yet exact"

def observation_embeddings51(glucose_unit=1, map_unit=2,
                             measurement_ekind=1, icu_ekind=4,
                             hospital_ekind=3):
    n = len(obs_codes51)
    nums = np.zeros((n, 6), dtype=np.float32)
    bins = {a: np.zeros(n, dtype=np.int64)
            for a in ("value", "dose", "rate")}
    for code in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        mask = (obs_codes51 == code) & np.isfinite(obs_values51)
        spec = num42[(code, "value")]
        v = obs_values51[mask]
        nums[mask, 0] = np.clip(
            (v.astype(np.float64) - spec["median"]) / spec["iqr"], -10, 10)
        nums[mask, 1] = 1
        bins["value"][mask] = (
            np.searchsorted(spec["fine_bin_edges"], v, side="left") + 1)
    unit_ids = np.zeros(n, dtype=np.int64)
    unit_ids[np.isin(obs_codes51, ["glucose_finger", "glucose_lab"])] = glucose_unit
    unit_ids[np.isin(obs_codes51, ["map_arterial", "map_cuff"])] = map_unit
    ek = np.full(n, measurement_ekind, dtype=np.int64)
    ek[obs_codes51 == "hospital_admission"] = hospital_ekind
    ek[obs_codes51 == "icu_admission"] = icu_ekind
    ek[np.isin(obs_codes51, ["insulin", "norepinephrine"])] = 2
    emb = (
        wt42("embed.code.weight")[
            torch.tensor([code42[c] for c in obs_codes51], dtype=torch.long)]
        + wt42("embed.kind.weight")[torch.tensor(obs_kinds51)]
        + wt42("embed.ekind.weight")[torch.tensor(ek)]
        + defaults42
        + wt42("embed.categorical.unit.weight")[torch.tensor(unit_ids)]
        + wt42("embed.categorical.rate_unit.weight")[0]
        + wt42("embed.kq.weight")[0]
    )
    for attr in ("value", "dose", "rate"):
        emb = emb + wt42(f"embed.bins.{attr}.weight")[
            torch.tensor(bins[attr], dtype=torch.long)]
    return emb + F.linear(torch.tensor(nums),
                          wt42("embed.numeric.weight"),
                          wt42("embed.numeric.bias"))

visit_bundles51 = np.unique(obs_owner51[
    np.isin(obs_codes51, ["hospital_admission", "icu_admission"])])
clean_mask51 = ~np.isin(valid_indices51, visit_bundles51)
clean_mask51 &= obs_counts51[valid_indices51] > 0

def evaluate_observations51(gu=1, mu=2, me=1, ie=4, he=3):
    e = encode_set50(observation_embeddings51(gu, mu, me, ie, he),
                     obs_owner51, obs_counts51, "encode")
    hidden = F.gelu(affine42(
        torch.cat([ctx51, e[valid_indices51], u_all51[valid_indices51]], dim=1),
        "z_post.0"))
    z = affine42(hidden, "z_post.2")[:, :32]
    err = z - target_z51
    rms = float(torch.sqrt(torch.mean(err[clean_mask51] ** 2)))
    return rms, e, hidden, z

initial51 = evaluate_observations51()
unit_results51 = [(initial51[0], (1, 2, 1))]
best_config51 = (1, 2, 1)
if initial51[0] > 1e-4:
    n_units52 = wt42("embed.categorical.unit.weight").shape[0]
    for gu52 in range(n_units52):
        for mu52 in range(n_units52):
            result52 = evaluate_observations51(gu52, mu52, 1)
            unit_results51.append((result52[0], (gu52, mu52, 1)))
    best_config51 = min(unit_results51)[1]
    if min(unit_results51)[0] > 1e-4:
        for me52 in range(wt42("embed.ekind.weight").shape[0]):
            result52 = evaluate_observations51(
                best_config51[0], best_config51[1], me52)
            unit_results51.append(
                (result52[0], (best_config51[0], best_config51[1], me52)))
        best_config51 = min(unit_results51)[1]
print("Best clean-bundle configurations:", sorted(unit_results51)[:5])

visit_results51 = []
best_all51 = None
for ie52 in range(wt42("embed.ekind.weight").shape[0]):
    result52 = evaluate_observations51(*best_config51, ie52, 3)
    total52 = float(torch.sqrt(torch.mean(
        (result52[3] - target_z51) ** 2)))
    visit_results51.append((total52, ie52))
    if best_all51 is None or total52 < best_all51[0]:
        best_all51 = (total52, ie52, *result52[1:])

e_all51 = best_all51[2]
hidden_all51 = best_all51[3]
print("Visit conventions:", sorted(visit_results51))
print("Whole-trajectory posterior RMS:", best_all51[0])
assert best_all51[0] < 1e-4, "Posterior reconstruction needs further correction"

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 51–52 / whole-trajectory accounting\n"
        "Run 51 stopped at bundle-count assertion without editing weights. "
        f"Run 52 added {added52} non-drug END markers; counts now exact. "
        f"Observation config {best_config51}, ICU ekind {best_all51[1]}; "
        f"all-bundle posterior RMS {best_all51[0]:.6g}. No edits.\n"
    )
```