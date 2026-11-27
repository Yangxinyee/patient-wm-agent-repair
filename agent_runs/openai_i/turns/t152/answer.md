The saved states align with the training records, and the model uses four attention heads. I’ll first validate atom encoding against the saved posterior means, including unit indices and empty-bundle handling. A sharply lower reconstruction error for one encoding will support it; otherwise, recurrent-response estimates would remain unreliable.

```python
numeric = {(s["code"], s["attribute"]): s for s in VOCAB["numeric"]}
event_kind_ids = {v: int(k) for k, v in VOCAB["event_kinds"].items()}
state_lookup = {(int(s), int(round(h * 60))): i
                for i, (s, h) in enumerate(zip(sid, hours))}
bundles = [{"starts": [], "ends": []} for _ in range(len(S))]

for row in records.to_dict("records"):
    key = (int(row["stay_id"]), int(round(row["hour"] * 60)))
    if key in state_lookup:
        bundles[state_lookup[key]]["starts"].append(row)
    if pd.notna(row["end_hour"]):
        end_key = (int(row["stay_id"]), int(round(row["end_hour"] * 60)))
        if end_key in state_lookup:
            bundles[state_lookup[end_key]]["ends"].append(row)

counts137 = np.array([len(b["starts"]) + len(b["ends"]) for b in bundles])
print("bundle count mismatches:",
      int(np.sum(counts137 != statefile136["n_atoms"])))
print("categorical row norms:",
      {n: W[n].norm(dim=1).numpy().round(3).tolist()
       for n in NAMES if n.startswith("embed.categorical.")})

def atom137(row, unit_index=0, rate_unit_index=0, end=False, end_ekind=None):
    event = row["event"]
    ekind = event_kind_ids[code_info[event]["event_kind"]]
    if end_ekind is not None:
        ekind = end_ekind
    x = (W["embed.code.weight"][codes[event]]
         + W["embed.kind.weight"][2 if end else 0]
         + W["embed.ekind.weight"][ekind]).clone()
    numbers = np.zeros(6, dtype=np.float32)
    for attr, offset in [("value", 0), ("dose", 2), ("rate", 4)]:
        value = np.nan if end else row.get(attr, np.nan)
        spec = numeric.get((event, attr))
        bin_id = 0
        if spec is not None and pd.notna(value):
            numbers[offset] = np.clip(
                (float(value) - spec["median"]) / max(spec["iqr"], 1e-8), -10, 10)
            numbers[offset + 1] = 1.0
            bin_id = 1 + int(np.searchsorted(spec["fine_bin_edges"],
                                           float(value), side="left"))
        x += W[f"embed.bins.{attr}.weight"][bin_id]
    x += linear136(torch.from_numpy(numbers), "embed.numeric")
    for attr in ["unit", "rate_unit", "action", "status", "route", "discharged_to"]:
        index = (unit_index if attr == "unit" else
                 rate_unit_index if attr == "rate_unit" else 0)
        x += W[f"embed.categorical.{attr}.weight"][index]
    return x

def pool137(atoms, prefix, zero_empty=False):
    if not atoms and zero_empty:
        return torch.zeros(192)
    if atoms:
        pooled = mlp136(torch.stack(atoms), prefix + ".mlp1").sum(0) / np.sqrt(len(atoms))
    else:
        pooled = torch.zeros(192)
    pooled = pooled + linear136(torch.tensor([np.log1p(len(atoms))],
                                             dtype=torch.float32), prefix + ".size")
    return norm136(mlp136(pooled, prefix + ".mlp2"), prefix + ".norm")

previous = np.arange(len(S)) - 1
previous[0] = -1
previous[1:][sid[1:] != sid[:-1]] = -1
valid = np.flatnonzero(previous >= 0)
hprev = torch.from_numpy(S[previous[valid], 192:960].reshape(-1, 4, 192))
dt137 = torch.from_numpy(
    ((hours[valid] - hours[previous[valid]]) * 3600).astype(np.float32))
hflow137 = hprev * torch.exp(
    -dt137[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :])
ctx137 = (torch.from_numpy(S[previous[valid], :192])
          + norm136(hflow137.mean(1), "core.out_norm")
          + mlp136(torch.log(dt137.clamp_min(1)).unsqueeze(-1) / 10, "gap"))
context = torch.zeros(len(S), 192)
context[valid] = ctx137

def posterior137(indices, e, u):
    return mlp136(torch.cat([context[indices], e, u], -1), "z_post")[:, :32]

print("\n=== SINGLE-MEASUREMENT UNIT CALIBRATION ===")
unit_tests137 = {}
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ix = [i for i in valid
          if not bundles[i]["ends"] and len(bundles[i]["starts"]) == 1
          and bundles[i]["starts"][0]["event"] == event][:24]
    if not ix:
        print(event, "no isolated examples")
        continue
    results = []
    for unit_index in range(6):
        e = torch.stack([pool137([atom137(bundles[i]["starts"][0], unit_index)],
                                "encode") for i in ix])
        for empty_action in ["zero", "encoded"]:
            u0 = pool137([], "encode_action", zero_empty=(empty_action == "zero"))
            u = u0.expand(len(ix), -1)
            pred = posterior137(ix, e, u).numpy()
            rms = float(np.sqrt(np.mean((pred - S[ix, 960:]) ** 2)))
            results.append((rms, unit_index, empty_action))
    results.sort()
    unit_tests137[event] = results
    print(event, "n =", len(ix), "best:", results[:3])

print("\n=== DRUG ROW EXAMPLES ===")
print(records[records.event.isin(["insulin", "norepinephrine"])]
      .groupby("event").head(1).to_dict("records"))
print("current edits:", edited())
```