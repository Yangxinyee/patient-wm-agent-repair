The post-hour-12 norepinephrine lag test is consistent with a negligible response: on independent stays, the conditional shifts were −0.059 ± 0.187 mmHg for doubling and −0.002 ± 0.022 for halving. The strong all-hours association means I should not claim a universal null effect.

The weight checks confirm that dose magnitude is removed while measurement inputs and absent bins are unchanged. I’ll now prepare a fresh rollout batch. Exact zero shifts would confirm the edit’s intended behavior; nonzero shifts would reveal a remaining dose-dependent route.

```python
import hashlib

all_tuning_ids = sorted(tuning["stay_id"].unique().tolist())
final_rollout_ids = [int(x) for x in all_tuning_ids[300:400]]
assert len(all_tuning_ids) == 400
assert len(final_rollout_ids) == 100
print("=== FRESH ROLLOUT VALIDATION IDS ===")
print(json.dumps(final_rollout_ids))

print("\n=== EXISTING FACTUAL VALIDATION ===")
print(json.dumps(audit_summary, default=str))

print("\n=== EXISTING EDITED ROLLOUT FILE ===")
dose_path = Path("/work/fwd_4/dose_edit_edited.csv")
if dose_path.exists():
    final_previous_rollout = pd.read_csv(dose_path)
    print("Shape:", final_previous_rollout.shape)
    print("Columns:", list(final_previous_rollout.columns))
    print(final_previous_rollout.head(3).to_string(index=False))

print("\n=== EDIT MANIFEST ===")
assert set(edited()) == {
    "embed.numeric.weight",
    "embed.bins.dose.weight",
    "embed.bins.rate.weight"
}
final_edit_hash = hashlib.sha256()
for name in sorted(edited()):
    w = load(name).detach().cpu().numpy()
    w0 = load_original(name).detach().cpu().numpy()
    final_edit_hash.update(name.encode())
    final_edit_hash.update(w.tobytes())
    print(name, "changed_entries:", int(np.count_nonzero(w != w0)),
          "shape:", w.shape)
print("Edit SHA256:", final_edit_hash.hexdigest())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 28: final validation preparation\n"
        "- NE joint-lag analysis has strong all-hours associations, so no "
        "claim of universal absence of effects is warranted.\n"
        "- In the evaluated hours 12–23, joint p=0.504; on independent "
        "stays p=0.755. Independent conditional contrasts: x2 -0.05898 "
        "(SE 0.18747), half -0.00205 (SE 0.02218) mmHg. "
        "These are not rollout truth.\n"
        "- Confirmed zero numeric dose/rate columns, identical positive "
        "dose/rate-bin rows, unchanged measurement columns and absent rows.\n"
        "- Preparing exact-rollout validation on the last 100 tuning stays.\n"
        "- Current edit SHA256: " + final_edit_hash.hexdigest() + "\n"
        "- No weight changes. Python runs: 28. Forward passes: 7.\n"
    )
```
