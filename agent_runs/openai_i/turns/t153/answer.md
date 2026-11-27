Bundle counts match exactly, but posterior reconstruction is not yet close enough: RMS errors are about 0.019, and the inferred glucose unit indices disagree. I’ll test bin-boundary and activation conventions before moving on. One consistent convention that lowers errors across sensors would resolve this; continued disagreement would reject the encoder reconstruction.

```python
print("numeric note:", VOCAB.get("numeric_note"))
print("absent-bin norms:",
      {a: float(W[f"embed.bins.{a}.weight"][0].norm())
       for a in ["value", "dose", "rate"]})
print("stored out consistency:",
      float(np.sqrt(np.mean(
          (norm136(torch.from_numpy(S[:, 192:960].reshape(-1, 4, 192)).mean(1),
                   "core.out_norm").numpy() - S[:, :192]) ** 2))))

print("\n=== RETAINED ENCODING NOTES ===")
shown138 = 0
for sec138 in sections135:
    if any(term in sec138.lower() for term in
           ["silu", "gelu", "unit mapping", "unit indices", "bin boundary",
            "encoder reconstruction", "posterior reconstruction"]):
        print(sec138[:550])
        shown138 += 1
        if shown138 >= 6:
            break

def atom138(row, unit_index=0, rate_unit_index=0, end=False,
            end_ekind=None, bin_side="left"):
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
                                           float(value), side=bin_side))
        x += W[f"embed.bins.{attr}.weight"][bin_id]
    x += linear136(torch.from_numpy(numbers), "embed.numeric")
    for attr in ["unit", "rate_unit", "action", "status", "route", "discharged_to"]:
        index = (unit_index if attr == "unit" else
                 rate_unit_index if attr == "rate_unit" else 0)
        x += W[f"embed.categorical.{attr}.weight"][index]
    return x

def mlp138(x, prefix, activation="gelu"):
    hidden = linear136(x, prefix + ".0")
    hidden = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[activation](hidden)
    return linear136(hidden, prefix + ".2")

probe_ix138 = {
    event: [i for i in valid
            if not bundles[i]["ends"] and len(bundles[i]["starts"]) == 1
            and bundles[i]["starts"][0]["event"] == event][:16]
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
}
results138 = []
for side138 in ["left", "right"]:
    for act138 in ["gelu", "silu", "relu"]:
        u0138 = norm136(mlp138(linear136(torch.zeros(1), "encode_action.size"),
                              "encode_action.mlp2", act138), "encode_action.norm")
        for event138, ix138 in probe_ix138.items():
            if not ix138:
                continue
            for unit138 in range(6):
                atoms138 = torch.stack([
                    atom138(bundles[i]["starts"][0], unit138, bin_side=side138)
                    for i in ix138])
                e138 = norm136(
                    mlp138(mlp138(atoms138, "encode.mlp1", act138)
                           + linear136(torch.tensor([np.log(2)], dtype=torch.float32),
                                       "encode.size"),
                           "encode.mlp2", act138),
                    "encode.norm")
                pred138 = posterior137(ix138, e138,
                                       u0138.expand(len(ix138), -1)).numpy()
                rms138 = float(np.sqrt(np.mean((pred138 - S[ix138, 960:]) ** 2)))
                results138.append((event138, rms138, side138, act138, unit138))

print("\n=== BOUNDARY / ACTIVATION RESULTS ===")
for event138 in probe_ix138:
    ranked138 = sorted([r for r in results138 if r[0] == event138],
                       key=lambda r: r[1])
    print(event138, ranked138[:4])
```