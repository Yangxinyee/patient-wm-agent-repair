Infusion END atoms appear in the state sequence, so ignoring them would mis-reconstruct later updates. I’ll first use single-insulin bundles without END atoms to identify the encoding. Agreement with the saved posterior means would validate the units, binning and latent-context calculation before tackling attention and memory.

```python
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)
rm_W = {
    n: torch.as_tensor(np.asarray(load_original(n)).copy(), dtype=torch.float32)
    for n in NAMES
}
def rm_lin(x, prefix):
    return F.linear(x, rm_W[prefix + ".weight"], rm_W.get(prefix + ".bias"))
def rm_ln(x, prefix):
    return F.layer_norm(x, (192,), rm_W[prefix + ".weight"],
                        rm_W[prefix + ".bias"], eps=1e-5)
def rm_mlp(x, prefix, activation="gelu"):
    a = rm_lin(x, prefix + ".0")
    a = F.gelu(a) if activation == "gelu" else F.silu(a)
    return rm_lin(a, prefix + ".2")
def rm_set(atom_vectors, prefix, activation="gelu"):
    n = atom_vectors.shape[1]
    if n == 0:
        return torch.zeros(atom_vectors.shape[0], 192)
    pooled = rm_mlp(atom_vectors, prefix + ".mlp1", activation).sum(1) / np.sqrt(n)
    size = torch.full((atom_vectors.shape[0], 1), float(np.log1p(n)))
    return rm_ln(rm_mlp(pooled + rm_lin(size, prefix + ".size"),
                       prefix + ".mlp2", activation), prefix + ".norm")

rm_normspec = {(a["code"], a["attribute"]): a for a in VOCAB["numeric"]}
rm_records = pd.read_csv("/data/train.csv", nrows=20000)
rm_states = {k: np.concatenate([np.asarray(state_files[j][k]) for j in (4, 5)])
             for k in state_files[4].keys()}
rm_lookup = {
    (int(r.stay_id), round(float(r.hour), 4), r.event): r
    for r in rm_records.itertuples(index=False)
}
rm_pure_idx = []
for i in range(1, len(rm_states["hour"])):
    if (rm_states["stay_id"][i] == rm_states["stay_id"][i-1]
        and rm_states["has_insulin"][i]
        and not rm_states["has_norepinephrine"][i]
        and rm_states["n_atoms"][i] == 1
        and (int(rm_states["stay_id"][i]),
             round(float(rm_states["hour"][i]), 4), "insulin") in rm_lookup):
        rm_pure_idx.append(i)
rm_pure_idx = np.asarray(rm_pure_idx[:48])
rm_rows = [rm_lookup[(int(rm_states["stay_id"][i]),
                     round(float(rm_states["hour"][i]), 4), "insulin")]
           for i in rm_pure_idx]
rm_rates = np.asarray([r.rate for r in rm_rows], dtype=np.float32)
rm_doses = np.asarray([r.dose for r in rm_rows], dtype=np.float32)
rm_prev = torch.tensor(rm_states["x"][rm_pure_idx - 1])
rm_target_z = torch.tensor(rm_states["x"][rm_pure_idx, 960:])
rm_dt = torch.tensor(np.round(
    (rm_states["hour"][rm_pure_idx] -
     rm_states["hour"][rm_pure_idx-1]) * 3600), dtype=torch.float32).reshape(-1, 1)
rm_hflow = rm_prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -rm_dt[:, :, None] * torch.exp(rm_W["core.flow.a"])[None, None, :])
rm_ctx = (rm_prev[:, :192] + rm_ln(rm_hflow.mean(1), "core.out_norm")
          + rm_mlp(torch.log1p(rm_dt)/10, "gap"))

def rm_insulin_atoms(unit_id, rate_unit_id, side, use_kq, activation):
    b = len(rm_rates)
    a = (rm_W["embed.code.weight"][8] + rm_W["embed.kind.weight"][0]
         + rm_W["embed.ekind.weight"][2]
         + rm_W["embed.categorical.unit.weight"][unit_id]
         + rm_W["embed.categorical.rate_unit.weight"][rate_unit_id])
    if use_kq:
        a = a + rm_W["embed.kq.weight"][0]
    a = a.repeat(b, 1)
    numeric = torch.zeros(b, 6)
    for attr, vals, zcol, flagcol in (
            ("dose", rm_doses, 2, 3), ("rate", rm_rates, 4, 5)):
        spec = rm_normspec[("insulin", attr)]
        numeric[:, zcol] = torch.tensor(np.clip(
            (vals-spec["median"])/spec["iqr"], -10, 10))
        numeric[:, flagcol] = 1
        bins = torch.tensor(
            1 + np.searchsorted(spec["fine_bin_edges"], vals, side=side),
            dtype=torch.long)
        a = a + rm_W[f"embed.bins.{attr}.weight"][bins]
    return (a + rm_lin(numeric, "embed.numeric")).unsqueeze(1)

rm_encoding_scores = []
for unit_id, rate_unit_id, side, use_kq, activation in itertools.product(
        range(2, 6), (2, 3), ("left", "right"), (False, True), ("gelu", "silu")):
    atoms = rm_insulin_atoms(unit_id, rate_unit_id, side, use_kq, activation)
    u = rm_set(atoms, "encode_action", activation)
    e = torch.zeros_like(u)
    z_pred = rm_mlp(torch.cat([rm_ctx, e, u], 1), "z_post", activation)[:, :32]
    err = float(torch.sqrt(torch.mean((z_pred-rm_target_z)**2)))
    rm_encoding_scores.append(
        (err, unit_id, rate_unit_id, side, use_kq, activation))
rm_encoding_scores.sort()
print("single-insulin bundles:", len(rm_pure_idx))
print("posterior target SD:", float(rm_target_z.std()))
print("best encodings (RMSE, unit, rate_unit, bin_side, kq, activation):")
for result in rm_encoding_scores[:10]:
    print(result)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Local-step encoding check; Python 89, forwards 11/50\n"
            "- Tested single-insulin bundles against saved original posterior means.\n"
            f"- Best encoding candidate: {rm_encoding_scores[0]}.\n"
            "- No additional weight edits; full state-update validation still required.\n")
```