The original rollout predicts modest shifts: about −2.8 for doubled insulin and +0.25 for doubled norepinephrine before reached-hour filtering. Those are much smaller than the provisional record-based targets. I’ll first verify that the exported slots reproduce `out`, and inspect predictions with no prior reading; these checks separate state-layout errors from register errors.

```python
r28_export = np.load("/work/fwd_3/states_original.npz")
r28_x = torch.tensor(r28_export["x"])
r28_out = r28_x[:, :192]
r28_h = r28_x[:, 192:960].reshape(-1, 4, 192)
r28_rebuilt = torch.nn.functional.layer_norm(
    r28_h.mean(1), (192,),
    torch.tensor(load_original("core.out_norm.weight")),
    torch.tensor(load_original("core.out_norm.bias"))
print("=== STATE IDENTITY ===")
print("out versus norm(mean(h)): RMSE",
      float((r28_out-r28_rebuilt).square().mean().sqrt()),
      "max", float((r28_out-r28_rebuilt).abs().max()))

print("\n=== OFFICIAL SUMMARIES ===")
for r28_k in (1, 2):
    r28_summary = json.load(open(f"/work/fwd_{r28_k}/summary.json"))
    print(r28_k, json.dumps(r28_summary)[:2600])
r28_roll = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("Reached-row shifts:",
      r28_roll.loc[r28_roll.reached].groupby("edit")["shift"].mean().to_dict())

print("\n=== EXACT SMALL VOCAB FIELDS ===")
print("VOCAB keys:", list(VOCAB))
for r28_key in ("config", "codes", "categorical", "event_kinds", "kinds"):
    if r28_key in VOCAB:
        print(r28_key, str(VOCAB[r28_key])[:1400])

print("\n=== PRIOR, QUERY, REGISTER AND VALUE HEAD TENSORS ===")
for r28_name in NAMES:
    if any(r28_name.startswith(s) for s in (
        "z_prior", "z_head", "type_emb", "last_numeric", "last_status",
        "query_norm", "attr_heads.value_bin", "gap."
    )):
        print(r28_name, SHAPES[r28_name])

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

with open("/work/notes.md", "a") as r28_note:
    r28_note.write(
        "\n## Run 28\n"
        "Original unfiltered rollout shifts are I x2 -2.770, I half +1.519, "
        "NE x2 +0.254, NE half -0.476, far below provisional causal targets. "
        "Checking out=norm(mean(h)) and no-last-reading cases to separate "
        "state/head errors from register-layout errors. No edits.\n"
    )
```