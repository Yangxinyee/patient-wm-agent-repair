The syntax error prevented all checks from running, so it provides no new evidence. I’ll rerun the corrected diagnostic. An exact state identity would confirm the layout; a mismatch would explain why the head reconstruction failed.

```python
r29_export = np.load("/work/fwd_3/states_original.npz")
r29_x = torch.tensor(r29_export["x"])
r29_out = r29_x[:, :192]
r29_h = r29_x[:, 192:960].reshape(-1, 4, 192)
r29_rebuilt = torch.nn.functional.layer_norm(
    r29_h.mean(1),
    normalized_shape=(192,),
    weight=torch.as_tensor(load_original("core.out_norm.weight")),
    bias=torch.as_tensor(load_original("core.out_norm.bias"))
)
print("=== STATE IDENTITY ===")
print("out versus norm(mean(h)): RMSE",
      float((r29_out-r29_rebuilt).square().mean().sqrt()),
      "max", float((r29_out-r29_rebuilt).abs().max()))

print("\n=== OFFICIAL SUMMARIES ===")
for r29_k in (1, 2):
    r29_summary = json.load(open(f"/work/fwd_{r29_k}/summary.json"))
    print(r29_k, json.dumps(r29_summary)[:2600])
r29_roll = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("Reached-row shifts:",
      r29_roll.loc[r29_roll.reached].groupby("edit")["shift"].mean().to_dict())

print("\n=== EXACT SMALL VOCAB FIELDS ===")
print("VOCAB keys:", list(VOCAB))
for r29_key in ("config", "codes", "categorical", "event_kinds", "kinds"):
    if r29_key in VOCAB:
        print(r29_key, str(VOCAB[r29_key])[:1400])

print("\n=== PRIOR, QUERY, REGISTER AND VALUE HEAD TENSORS ===")
for r29_name in NAMES:
    if any(r29_name.startswith(s) for s in (
        "z_prior", "z_head", "type_emb", "last_numeric", "last_status",
        "query_norm", "attr_heads.value_bin", "gap."
    )):
        print(r29_name, SHAPES[r29_name])

print("\n=== RECONSTRUCTION SAMPLE ===")
print(ds22.head(6).to_string(index=False))
print("pre indices:", pre22[:6], "dt:", dt22[:6])
print("has_last:", has22[:6], "last_z:", last_z22[:6],
      "last_bin:", last_bin22[:6].tolist(), "age_hours:", age_hours22[:6])
print("Register table columns:", reg22.columns.tolist())
print(reg22.head(3).to_string(index=False))

print("\n=== FIRST-READING COUNTS ===")
print(f22.loc[f22.hour == -1].groupby("event").size().to_dict())
print("First-reading mean prediction:",
      f22.loc[f22.hour == -1].groupby("event").predicted_mean.mean().to_dict())

with open("/work/notes.md", "a") as r29_note:
    r29_note.write(
        "\n## Runs 28–29\n"
        "Run 28 had a syntax error and executed nothing. Corrected diagnostics "
        "in run 29: state identity, official reached-row summaries, exact head "
        "tensor dimensions, and no-last-reading cases. No edits.\n"
    )
```