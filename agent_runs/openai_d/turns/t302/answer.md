The lookup indexing is confirmed. The diagnostic stopped because the cached data are dictionaries; it changed no weights. I’ll verify which correction sign reproduces the earlier results, then install the heavily shrunk, scale-4 correction. The next factual pass will test whether its apparent gain holds on the previously failing cohort.

```python
r257_g = dict(globals())
r257_tune_names = [
    n for n, v in r257_g.items()
    if n.startswith("c210_") and isinstance(v, dict)
    and "y" in v and len(v["y"]) == len(r252_base["tune"])
]
assert r257_tune_names, "Tuning cache not found; no edit installed."
r257_tune_name = sorted(r257_tune_names)[0]
r257_caches = {"train": c210_train, "tune": r257_g[r257_tune_name]}
print("Tuning cache:", r257_tune_name)

r257_orig_ix = {}
for split, cache in r257_caches.items():
    errors = [
        float(np.max(np.abs(r252_nll(L, cache) - r252_base[split])))
        for L in r252_logits[split]
    ]
    r257_orig_ix[split] = int(np.argmin(errors))
    assert min(errors) < 1e-5
    print(split, "baseline-match errors:", errors)

r257_table = r252_tables[80.0]
r257_expected = r254_screen[
    (r254_screen["shrink"] == 80.0) & (r254_screen["scale"] == 4.0)
].iloc[0]
r257_checks = []
for sign in (1.0, -1.0):
    deltas = {}
    for split, cache in r257_caches.items():
        L = r252_logits[split][1 - r257_orig_ix[split]]
        correction = sign * 4.0 * r252_table_rows(r257_table, cache)
        deltas[split] = r252_nll(L, cache, correction) - r252_base[split]
    row = {
        "sign": sign,
        "train_fit": float(deltas["train"][r252_fit].mean()),
        "train_validation": float(deltas["train"][r252_val].mean()),
        "tune100": float(deltas["tune"].mean()),
    }
    row["reproduction_error"] = max(
        abs(row[k] - float(r257_expected[k]))
        for k in ("train_fit", "train_validation", "tune100")
    )
    r257_checks.append(row)
print(pd.DataFrame(r257_checks).to_string(index=False))
r257_best = min(r257_checks, key=lambda row: row["reproduction_error"])
assert r257_best["reproduction_error"] < 2e-6, "Screen mismatch; no edit installed."
r257_sign = r257_best["sign"]

# The screened head must still be the current head.
for n, v in r252_head.items():
    assert np.allclose(
        torch.as_tensor(load(n)).numpy(),
        torch.as_tensor(v).numpy(), atol=1e-7, rtol=0
    ), f"Head changed since screen: {n}"

r257_transition_name = "embed_transition.value_bin.weight"
assert r257_transition_name not in edited()
r257_transition = torch.as_tensor(
    load_original(r257_transition_name), dtype=torch.float32
).clone()
r257_codes = {item["name"]: item["id"] for item in VOCAB["codes"]}
for (event, last_bin), correction in r257_table.items():
    row = r257_codes[event] * 17 + int(last_bin)
    r257_transition[row] += torch.as_tensor(
        r257_sign * 4.0 * np.asarray(correction), dtype=torch.float32
    )

save_edit(r257_transition_name, r257_transition)
r257_checkpoint = {
    n: torch.as_tensor(load(n)).clone() for n in edited()
}
np.savez("/work/lastbin80_scale4.npz",
         **{n: v.numpy() for n, v in r257_checkpoint.items()})
print("Installed edits:", edited())
print("Cached deltas:", {k: r257_best[k] for k in
                         ("train_fit", "train_validation", "tune100")})

print("\n=== CONTINUED EARLY CAUSAL FINDINGS ===")
print(r255_notes[5800:10300])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 256–257: previous-bin cancellation installed\n"
            "Run 256's cache inspection stopped on dictionary attribute access; "
            "no weights were changed by that run. Run 257 reproduced the shrink80, "
            "scale4 screen and installed its correction in "
            "embed_transition.value_bin.weight, keeping the validated .94 insulin/"
            ".50 NE head unchanged. Coefficients were fitted only on training "
            "records. GPU factual fit and rollout preservation remain unverified.\n"
            + str(r257_best) + "\n")
```