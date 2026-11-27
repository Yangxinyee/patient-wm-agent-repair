Half and double doses substantially change the posterior, so the bottleneck is not simply missing dose information at the input. The retained causal kernels also show delayed effects, especially for insulin.

I’ll reconstruct measurement-bundle encodings next so we can trace where the recurrent model loses that information. Near-exact agreement with exported posteriors would validate this reconstruction; a larger mismatch would make downstream edit calculations unreliable.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

def atom_embedding_repair(event, values=None, rates=None, doses=None,
                          unit=0, rate_unit=0, kind=0, ekind=None,
                          side="left"):
    if ekind is None:
        ekind = 2 if event in ("insulin", "norepinephrine") else (
            1 if event in (
                "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"
            ) else 3
        )
    provided = [v for v in (values, rates, doses) if v is not None]
    n = len(np.atleast_1d(provided[0])) if provided else 1
    nums = np.zeros((n, 6), dtype=np.float32)
    bin_indices = {}
    for attr, supplied, zcol, flagcol in [
        ("value", values, 0, 1), ("dose", doses, 2, 3),
        ("rate", rates, 4, 5)
    ]:
        bins = np.zeros(n, dtype=np.int64)
        if supplied is not None:
            raw = np.asarray(supplied, dtype=float).reshape(-1)
            valid = np.isfinite(raw)
            spec = numeric_specs_repair[(event, attr)]
            nums[valid, zcol] = np.clip(
                (raw[valid] - spec["median"]) / spec["iqr"], -10, 10
            )
            nums[valid, flagcol] = 1
            bins[valid] = 1 + np.searchsorted(
                spec["fine_bin_edges"], raw[valid], side=side
            )
        bin_indices[attr] = bins

    constant = (
        rW["embed.code.weight"][code_ids_repair[event]]
        + rW["embed.kind.weight"][kind]
        + rW["embed.ekind.weight"][ekind]
        + rW["embed.kq.weight"][0]
        + rW["embed.text.weight"][0]
    )
    for name in NAMES:
        if name.startswith("embed.categorical.") and name.endswith(".weight"):
            category = name.split(".")[2]
            index = unit if category == "unit" else (
                rate_unit if category == "rate_unit" else 0
            )
            constant = constant + rW[name][index]

    atom = rp_linear(torch.from_numpy(nums), "embed.numeric") + constant
    for attr, bins in bin_indices.items():
        atom = atom + rW[f"embed.bins.{attr}.weight"][bins]
    return atom

print("=== FULL ATOM CHECK AGAINST VERIFIED DRUG ATOMS ===")
for r45_drug in ("insulin", "norepinephrine"):
    r45_fixture = controlled_action_repair[r45_drug]["records"]
    r45_unit, r45_runit, _, r45_side = atom_recipe_repair[r45_drug]
    r45_atoms = atom_embedding_repair(
        r45_drug,
        rates=r45_fixture["rate"].to_numpy(),
        doses=r45_fixture["dose"].to_numpy(),
        unit=r45_unit, rate_unit=r45_runit, side=r45_side
    )
    print(r45_drug, "atom RMSE:", float(torch.mean(
        (r45_atoms - action_atom_vectors_repair[r45_drug]) ** 2
    ).sqrt()))

r45_bank = state_train_repair
r45_stays = np.unique(r45_bank["stay_id"])
r45_records = train.loc[
    train["stay_id"].isin(r45_stays)
].copy()
r45_records["minute"] = np.rint(r45_records["hour"] * 60).astype(int)
r45_state_frame = pd.DataFrame({
    "state_index": np.arange(len(r45_bank["x"])),
    "stay_id": r45_bank["stay_id"],
    "minute": np.rint(r45_bank["hour"] * 60).astype(int),
    "n_atoms": r45_bank["n_atoms"],
    "has_insulin": r45_bank["has_insulin"],
    "has_norepinephrine": r45_bank["has_norepinephrine"]
})
r45_measure_fixture = r45_state_frame.merge(
    r45_records, on=["stay_id", "minute"], how="inner"
)
r45_measure_fixture = r45_measure_fixture.loc[
    (r45_measure_fixture["n_atoms"] == 1)
    & ~r45_measure_fixture["has_insulin"]
    & ~r45_measure_fixture["has_norepinephrine"]
    & r45_measure_fixture["event"].isin(
        ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
    )
    & (r45_measure_fixture["state_index"] > 0)
].copy()
r45_indices = r45_measure_fixture["state_index"].to_numpy()
r45_same_stay = (
    r45_bank["stay_id"][r45_indices]
    == r45_bank["stay_id"][r45_indices - 1]
)
r45_measure_fixture = r45_measure_fixture.loc[r45_same_stay].copy()

def pre_context_repair(bank, indices):
    previous = torch.as_tensor(
        bank["x"][indices - 1], dtype=torch.float32
    )
    minute = np.rint(bank["hour"] * 60).astype(int)
    dt = torch.as_tensor(
        (minute[indices] - minute[indices - 1]) * 60,
        dtype=torch.float32
    )
    h = previous[:, 192:960].reshape(-1, 4, 192)
    h = h * torch.exp(
        -dt[:, None, None] * torch.exp(rW["core.flow.a"])[None, None, :]
    )
    context = (
        previous[:, :192]
        + rp_norm(h.mean(dim=1), "core.out_norm")
        + rp_mlp(torch.log1p(dt)[:, None] / 10, "gap")
    )
    return context, h, dt

print("\n=== SINGLE-MEASUREMENT POSTERIOR CHECK ===")
measurement_recipe_repair = {}
for r45_event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    r45_fixture = r45_measure_fixture.loc[
        r45_measure_fixture["event"] == r45_event
    ].head(100)
    if len(r45_fixture) == 0:
        print(r45_event, "no singleton fixtures")
        continue
    r45_idx = r45_fixture["state_index"].to_numpy()
    r45_ctx, _, _ = pre_context_repair(r45_bank, r45_idx)
    r45_target = torch.as_tensor(
        r45_bank["x"][r45_idx, 960:992], dtype=torch.float32
    )
    r45_tests = []
    for r45_unit in range(6):
        for r45_side in ("left", "right"):
            r45_atoms = atom_embedding_repair(
                r45_event, values=r45_fixture["value"].to_numpy(),
                unit=r45_unit, side=r45_side
            )
            r45_e = rp_norm(
                rp_mlp(
                    rp_mlp(r45_atoms, "encode.mlp1")
                    + rp_linear(
                        torch.full((len(r45_atoms), 1), float(np.log(2))),
                        "encode.size"
                    ),
                    "encode.mlp2"
                ),
                "encode.norm"
            )
            r45_z = rp_mlp(torch.cat([
                r45_ctx, r45_e, torch.zeros_like(r45_e)
            ], dim=-1), "z_post")[:, :32]
            r45_error = float(torch.mean((r45_z - r45_target) ** 2).sqrt())
            r45_tests.append((r45_error, r45_unit, r45_side))
    r45_tests.sort()
    measurement_recipe_repair[r45_event] = (
        r45_tests[0][1], r45_tests[0][2]
    )
    print(r45_event, "n", len(r45_fixture), "best:", r45_tests[:2])

print("measurement recipes:", measurement_recipe_repair)
print("current edits:", edited())

with open("/work/notes.md", "a") as r45_file:
    r45_file.write(
        "\n## Run 45\n"
        "Controlled dose changes move action encodings and posterior means substantially, "
        "so the input/posterior is not simply dose-insensitive. Implemented general atom "
        "embedding and tested singleton measurement encodings against exported posterior "
        "means as a prerequisite for reconstructing memory and recurrent transitions. No edits.\n"
    )
```
