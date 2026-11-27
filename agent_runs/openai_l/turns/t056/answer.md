Infusion ends belong in the observation encoder, with missing numeric attributes. That reproduces all drug-bundle posteriors to numerical precision. I’ll extend the reconstruction to ordinary reading bundles so the correction can be constrained on the whole factual trajectory, not just drug events.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

print("Event kinds:", VOCAB["event_kinds"])
print("Unit embedding size:", wt42("embed.categorical.unit.weight").shape[0])

s51 = s49
valid_indices51 = np.flatnonzero(
    np.r_[False, s51["stay_id"][1:] == s51["stay_id"][:-1]]
)
ctx51 = contexts37(s51, valid_indices51)
target_z51 = torch.tensor(s51["x"][valid_indices51, 960:],
                          dtype=torch.float32)
all_keys51 = {
    (int(s51["stay_id"][i]), int(round(float(s51["hour"][i]) * 10000))): i
    for i in range(len(s51["hour"]))
}
world51 = train[
    train.stay_id.isin(np.unique(s51["stay_id"]))
    & ~train.event.isin(["insulin", "norepinephrine"])
].copy()

obs_meta51 = []
for row51 in world51.itertuples(index=False):
    key51 = (int(row51.stay_id), int(round(row51.hour * 10000)))
    if key51 in all_keys51:
        obs_meta51.append(
            (all_keys51[key51], row51.event, float(row51.value), 0)
        )
for row51 in drug_records49.itertuples(index=False):
    key51 = (int(row51.stay_id), int(round(row51.end_hour * 10000)))
    if key51 in all_keys51:
        obs_meta51.append((all_keys51[key51], row51.event, np.nan, 2))

obs_array51 = np.asarray(obs_meta51, dtype=object)
obs_owner51 = obs_array51[:, 0].astype(np.int64)
obs_codes51 = obs_array51[:, 1].astype(str)
obs_values51 = obs_array51[:, 2].astype(np.float32)
obs_kinds51 = obs_array51[:, 3].astype(np.int64)
obs_counts51 = np.bincount(obs_owner51, minlength=len(s51["hour"]))

u_all51 = torch.zeros((len(s51["hour"]), 192), dtype=torch.float32)
u_all51[drug_indices49] = u_starts50
action_counts51 = np.zeros(len(s51["hour"]), dtype=np.int64)
action_counts51[drug_indices49] = start_counts50
assert np.array_equal(obs_counts51 + action_counts51, s51["n_atoms"])

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
    np.isin(obs_codes51, ["hospital_admission", "icu_admission"])
])
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
print("Expected-unit clean-bundle RMS:", initial51[0])
unit_results51 = [(initial51[0], (1, 2, 1))]
best_config51 = (1, 2, 1)
if initial51[0] > 1e-4:
    n_units51 = wt42("embed.categorical.unit.weight").shape[0]
    for gu51 in range(n_units51):
        for mu51 in range(n_units51):
            result51 = evaluate_observations51(gu51, mu51, 1)
            unit_results51.append((result51[0], (gu51, mu51, 1)))
    best_config51 = min(unit_results51)[1]
    if min(unit_results51)[0] > 1e-4:
        for me51 in range(wt42("embed.ekind.weight").shape[0]):
            result51 = evaluate_observations51(
                best_config51[0], best_config51[1], me51)
            unit_results51.append(
                (result51[0], (best_config51[0], best_config51[1], me51)))
        best_config51 = min(unit_results51)[1]
print("Best clean-bundle configurations:", sorted(unit_results51)[:5])

visit_results51 = []
best_all51 = None
for ie51 in range(wt42("embed.ekind.weight").shape[0]):
    rms51, e51, hidden51, z51 = evaluate_observations51(
        *best_config51, ie51, 3)
    total51 = float(torch.sqrt(torch.mean((z51 - target_z51) ** 2)))
    visit_results51.append((total51, ie51))
    if best_all51 is None or total51 < best_all51[0]:
        best_all51 = (total51, ie51, e51, hidden51, z51)

e_all51 = best_all51[2]
hidden_all51 = best_all51[3]
print("Visit conventions:", sorted(visit_results51))
print("Whole-trajectory posterior RMS:", best_all51[0])
for label51, mask51 in (
    ("reading bundles", (s51["hour"][valid_indices51] % 1) < 0.01),
    ("drug bundles", np.isin(valid_indices51, drug_indices49))
):
    err51 = best_all51[4][mask51] - target_z51[mask51]
    print(label51, int(mask51.sum()),
          float(torch.sqrt(torch.mean(err51 ** 2))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 51 / whole-trajectory posterior reconstruction\n"
        "END atoms are observation atoms: kind 2, event-kind 2, no numeric "
        "attributes, unit/rate-unit 0, action category 0. "
        f"Observation configuration {best_config51}, ICU ekind {best_all51[1]}; "
        f"whole-trajectory posterior RMS {best_all51[0]:.6g}. "
        "No tensors edited.\n"
    )
```