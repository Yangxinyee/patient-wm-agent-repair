The tool returned 4,313 states from 100 training stays. I’ll align them with infusion records and reconstruct the pre-action context. I’ll also match the reference shifts to the rollout’s reached reading rows, since the earlier averages were approximate. Accurate alignment and reproducible state calculations would let us edit the dose pathway rather than unrelated prediction heads.

```python
torch.set_num_threads(8)
Sfile = np.load("/work/fwd_3/states_original.npz")
S = {k: Sfile[k] for k in Sfile.files}
print("STATE ARRAYS:", {k: (v.shape, str(v.dtype)) for k, v in S.items()})
print(pd.DataFrame({k: v for k, v in S.items() if k != "x"}).head(24).to_string(index=False))
print("fractional hours:", np.unique(np.round(S["hour"] % 1, 5), return_counts=True))

existing_frames = [(k, v) for k, v in list(globals().items())
                   if isinstance(v, pd.DataFrame) and
                   {"stay_id", "event", "hour", "rate"}.issubset(v.columns)]
print("record dataframes:", [(k, v.shape) for k, v in existing_frames])
if existing_frames:
    candidates = [v for k, v in existing_frames if v["stay_id"].min() < 2000000000]
else:
    candidates = []
if candidates:
    state_records = candidates[0][candidates[0]["stay_id"].isin(np.unique(S["stay_id"]))].copy()
else:
    parts = []
    for chunk in pd.read_csv("/data/train.csv", chunksize=250000):
        parts.append(chunk[chunk.stay_id.isin(np.unique(S["stay_id"]))])
    state_records = pd.concat(parts, ignore_index=True)
print("\nRECORD SAMPLE:")
print(state_records.head(24).to_string(index=False))

W = {name: torch.as_tensor(load_original(name), dtype=torch.float32).clone() for name in NAMES}
def linear(t, prefix):
    return torch.nn.functional.linear(t, W[prefix + ".weight"], W.get(prefix + ".bias"))
def layer_norm(t, prefix):
    return torch.nn.functional.layer_norm(t, (t.shape[-1],),
                                         W[prefix + ".weight"], W[prefix + ".bias"])
def mlp(t, prefix, activation="gelu"):
    t = linear(t, prefix + ".0")
    if activation == "gelu":
        t = torch.nn.functional.gelu(t)
    elif activation == "relu":
        t = torch.relu(t)
    else:
        t = torch.nn.functional.silu(t)
    return linear(t, prefix + ".2")

sx = torch.tensor(S["x"], dtype=torch.float32)
prev_idx = np.arange(len(sx)) - 1
same_stay = np.r_[False, S["stay_id"][1:] == S["stay_id"][:-1]]
drugs_mask = S["has_insulin"].astype(bool) | S["has_norepinephrine"].astype(bool)
drug_idx = np.where(same_stay & drugs_mask)[0]
dt_drug = torch.tensor((S["hour"][drug_idx] - S["hour"][drug_idx-1]) * 3600,
                       dtype=torch.float32)
h_prev = sx[drug_idx-1, 192:960].reshape(-1, 4, 192)
h_flow = h_prev * torch.exp(-dt_drug[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :])
base_ctx = sx[drug_idx-1, :192] + layer_norm(h_flow.mean(1), "core.out_norm")
for activation in ["gelu", "relu", "silu"]:
    context = base_ctx + mlp(torch.log(dt_drug.clamp_min(1))[:, None] / 10,
                            "gap", activation)
    print("context", activation, "mean norm:", context.norm(dim=1).mean().item())
print("\nWEIGHT NORMS:")
for name in ["embed.numeric.weight", "z_post.0.weight", "z_post.2.weight",
             "z_in.weight", "encode_action.norm.weight", "core.flow.a"]:
    arrw = W[name].numpy()
    print(name, "norm", float(np.linalg.norm(arrw)), "range", arrw.min(), arrw.max())
print("event kinds:", VOCAB["event_kinds"], "atom kinds:", VOCAB["atom_kinds"])

print("\nREFERENCES WITH TOOL'S READING-ROW AVERAGING:")
matched_refs = {}
for edit, drug, sensor, mult in [
    ("insulin_x2", "insulin", "glucose_finger", 1.0),
    ("insulin_half", "insulin", "glucose_finger", -0.5),
    ("ne_x2", "norepinephrine", "map_arterial", 1.0),
    ("ne_half", "norepinephrine", "map_arterial", -0.5),
]:
    rr = roll0[(roll0.edit == edit) & roll0.reached.astype(bool)]
    ii = np.searchsorted(tuning_ids, rr.stay_id.to_numpy())
    hh = rr.hour.to_numpy(dtype=int) - 13
    vals = mult * kernel_refs[(drug, sensor)][0][ii, hh]
    matched_refs[edit] = float(vals.mean())
    print(edit, "estimated target:", vals.mean(), "original:", rr["shift"].mean(),
          "n:", len(rr))

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs: 10; forwards: 3. Original training states acquired "
            "(100 stays, 4313 bundles). Record-derived reference shifts matched "
            "to original rollout reading rows: " + json.dumps(matched_refs) + ".\n")
```
