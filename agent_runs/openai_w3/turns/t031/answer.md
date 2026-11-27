Neither the END search nor the numerical tests produced an exact match. Before relying on any local computation, I’ll verify that the cached weights still equal the original file and check the state’s documented normalization identity. A mismatch there would explain why the encoding searches failed.

```python
from safetensors.torch import load_file

disk_weights = load_file("/model/model.safetensors", device="cpu")
weight_differences = []
for name in NAMES:
    delta = float(torch.max(torch.abs(
        torch.as_tensor(W[name]) - disk_weights[name].float())))
    if delta > 1e-7:
        weight_differences.append((name, delta))
print("CACHED WEIGHTS DIFFERING FROM ORIGINAL:", weight_differences)
print("submitted edits:", edited())

xx_check = torch.as_tensor(np.asarray(S["x"]), dtype=torch.float32)
out_check = layer_norm(xx_check[:, 192:960].reshape(-1, 4, 192).mean(1), "core.out_norm")
print("out = norm(mean(h)) MSE:", float(((out_check-xx_check[:, :192])**2).mean()))
print("out identity max error:", float(torch.max(torch.abs(out_check-xx_check[:, :192]))))

print("\n=== POSTERIOR FIRST-LAYER BLOCK NORMS ===")
for label, lo, hi in (("context", 0, 192), ("observations", 192, 384), ("actions", 384, 576)):
    ww = W["z_post.0.weight"][:, lo:hi]
    print(label, "Frobenius", float(torch.linalg.vector_norm(ww)),
          "maximum", float(torch.max(torch.abs(ww))))

print("\n=== SAMPLE ALIGNMENT ===")
for event in ("map_cuff", "glucose_finger", "insulin", "insulin END"):
    for i in isolated_by_event[event][:2]:
        sid, hour = int(S["stay_id"][i]), float(S["hour"][i])
        here = state_records[
            (state_records.stay_id == sid) &
            ((np.abs(state_records.hour-hour) < 0.0002) |
             (np.abs(state_records.end_hour-hour) < 0.0002))
        ]
        print(event, "index", i, "stay", sid, "hour", hour,
              "n_atoms", np.asarray(S["n_atoms"])[i])
        print(here[["hour", "event", "value", "rate", "dose", "end_hour"]].to_string(index=False))

print("\n=== VALUE NUMERIC ORDER / ABSENCE CHECK ===")
for event in ("map_cuff", "glucose_finger"):
    idx = np.asarray(isolated_by_event[event][:12], dtype=int)
    records = [isolated_record(i, event) for i in idx]
    atoms = torch.stack([make_atom(r) for r in records])
    target = xx_check[idx, 960:]
    table = numeric_tables[(event, "value")]
    zv = torch.as_tensor(
        np.clip((np.array([r.value for r in records])-table["median"])/table["iqr"], -10, 10),
        dtype=torch.float32)
    scores = []
    for unit in range(6):
        aa = atoms + W["embed.categorical.unit.weight"][unit] - W["embed.categorical.unit.weight"][2 if event.startswith("map_") else 5]
        for label, bb in [
            ("standard", aa),
            ("swapped", aa + (1-zv)[:, None]*(W["embed.numeric.weight"][:, 0]-W["embed.numeric.weight"][:, 1])),
            ("no_value_numbers", aa-zv[:, None]*W["embed.numeric.weight"][:, 0]-W["embed.numeric.weight"][:, 1]),
        ]:
            for omit_kq in (False, True):
                cc = bb-W["embed.kq.weight"][0] if omit_kq else bb
                ee = encode_one_set(cc, "encode")
                zz = posterior_mean(state_context(idx), ee, torch.zeros_like(ee))
                scores.append((float(((zz-target)**2).mean()), (unit, label, omit_kq)))
    print(event, sorted(scores)[:4])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 27; forwards 3. No exact match from metadata, flow/input scale or END searches. Verifying cached weights, state layout and numeric ordering before choosing an edit mechanism. No submitted edits.\n")
```