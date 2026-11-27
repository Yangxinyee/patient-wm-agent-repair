All four factual batches stayed below the limit; their approximate pooled increase is 0.00337. I’ll calculate the exact result and stay-bootstrap uncertainty, then select additional stays for a separate safety check. An upper interval below 0.005 would support retaining the edit, but not guarantee performance on unseen stays.

```python
import json

assert set(edited()) == set(amplitude093_checkpoint)
assert all(np.array_equal(np.asarray(load(n)), amplitude093_checkpoint[n])
           for n in edited())

factual_pairs093 = []
for pass_number in (41, 44, 45, 46):
    base = Path(f"/work/fwd_{pass_number}")
    original = pd.read_csv(base / "factual_original.csv")
    current = pd.read_csv(base / "factual_edited.csv")
    pair = original.merge(
        current, on=["stay_id", "hour", "event"],
        suffixes=("_original", "_edited"), validate="one_to_one")
    assert len(pair) == len(original) == len(current)
    assert np.allclose(pair.observed_original, pair.observed_edited)
    pair["delta_nll"] = pair.nll_edited - pair.nll_original
    pair["forward_pass"] = pass_number
    factual_pairs093.append(pair)
factual400_093 = pd.concat(factual_pairs093, ignore_index=True)
assert factual400_093.stay_id.nunique() == 400

stay_cost093 = factual400_093.groupby("stay_id").agg(
    delta_sum=("delta_nll", "sum"), n=("delta_nll", "size"))
bootstrap_rng093 = np.random.default_rng(2902026)
bootstrap_indices093 = bootstrap_rng093.integers(
    0, len(stay_cost093), size=(12000, len(stay_cost093)))
bootstrap_cost093 = (
    stay_cost093.delta_sum.to_numpy()[bootstrap_indices093].sum(axis=1) /
    stay_cost093.n.to_numpy()[bootstrap_indices093].sum(axis=1))
cost_ci093 = np.quantile(bootstrap_cost093, [0.025, 0.975])

print("=== COMPLETE 400-STAY FACTUAL VALIDATION ===")
print("Readings:", len(factual400_093))
print("Original NLL:", factual400_093.nll_original.mean())
print("Edited NLL:", factual400_093.nll_edited.mean())
print("Paired increase:", factual400_093.delta_nll.mean())
print("Stay-bootstrap 95% interval:", cost_ci093.tolist())
print("By event:")
print(factual400_093.groupby("event").delta_nll.agg(
    ["size", "mean"]).to_string())

print("=== ROLLOUT FILE SCHEMA ===")
dose42_093 = pd.read_csv("/work/fwd_42/dose_edit_edited.csv")
print(dose42_093.columns.tolist())
print(dose42_093.head(2).to_string(index=False))

seen_forward_ids = set()
for path in Path("/work").glob("fwd_*/*original.csv"):
    header = pd.read_csv(path, nrows=0).columns
    if "stay_id" in header:
        seen_forward_ids.update(
            pd.read_csv(path, usecols=["stay_id"]).stay_id.unique().tolist())
train_id_pool = pd.read_csv(
    "/data/train.csv", usecols=["stay_id"]).stay_id.unique()
fresh_train_pool = np.array(
    [sid for sid in train_id_pool if sid not in seen_forward_ids])
fresh_train200 = np.random.default_rng(471930).choice(
    fresh_train_pool, size=200, replace=False)
validation_train_batches093 = [
    sorted(map(int, fresh_train200[:100])),
    sorted(map(int, fresh_train200[100:]))]
print("=== FRESH TRAINING-STAY SAFETY BATCHES ===")
print("BATCH A:", json.dumps(validation_train_batches093[0]))
print("BATCH B:", json.dumps(validation_train_batches093[1]))

print("=== IDENTIFICATION NOTEBOOK EXCERPT ===")
identification_start = notes_text.find(
    "## Recovery and threshold identification")
if identification_start >= 0:
    print(notes_text[identification_start:identification_start + 2700])
print("Matched causal targets:",
      str(globals().get("matched_causal_targets163"))[:900])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Amplitude093 complete factual validation; Python 290, forwards 46/50\n"
        f"- Exact full400 NLL increase {factual400_093.delta_nll.mean():.8f}"
        f" over {len(factual400_093)} readings; stay-bootstrap 95% interval"
        f" {cost_ci093.tolist()}.\n"
        "- First100 rollout insulin x2 -8.450, half +3.805; "
        "last100 -7.133, +3.107. NE remains close to original.\n"
        "- Current tensors exactly equal amplitude093_checkpoint; no weights changed.\n"
        "- Selected 200 additional training stays not previously forwarded, "
        "without inspecting their outcomes, for separate factual safety checks.\n"
        "- Remaining rollout batches will improve readout precision; "
        "unseen-stay factual safety remains an inference, not a guarantee.\n")
```