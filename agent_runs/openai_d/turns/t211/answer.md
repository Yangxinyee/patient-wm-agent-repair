Adding the transition logits removed nearly all the mismatch and recovered the earlier loss estimate. The guarded, stronger NE trial has a local overall loss increase of 0.00339, though its held-out estimate is borderline. I’ll install it for GPU testing. It would be supported if factual loss stays within 0.005 and NE readouts move toward the record-derived targets without disturbing insulin.

```python
assert c182_error < 1e-3, "Candidate reconstruction remains inconsistent."
assert np.isfinite(c182_candidate_logits).all()

c183_fallback = {
    name: c180_np(load(name)).copy()
    for name in r171_combined_backup
}
c183_candidate = {
    name: c180_np(tensor).astype(np.float32, copy=True)
    for name, tensor in r178_candidate.items()
}

for name, tensor in c183_candidate.items():
    assert tuple(tensor.shape) == tuple(SHAPES[name])
    assert np.isfinite(tensor).all()
    save_edit(name, tensor)

c183_hash = hashlib.sha256(
    b"".join(c183_candidate[name].tobytes()
             for name in sorted(c183_candidate))
).hexdigest()[:16]

print("Installed guarded NE trial:", c183_hash)
print("NE output multiplier:", r178_strength)
print("Retained insulin correction unchanged.")
print("Local overall/fit/validation loss deltas:",
      c182_rows[1]["all"],
      c182_rows[1]["fit"],
      c182_rows[1]["validation"])
print("Fallback preserved in c183_fallback and r171_combined_backup.")

for name in c183_candidate:
    assert np.array_equal(c180_np(load(name)), c183_candidate[name])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 183: guarded stronger-NE trial installed\n"
            "The run-178 discrepancy was primarily an omitted register "
            "transition-logit term, not unintended weight changes. Adding it "
            f"reduces reconstruction error to {c182_error:.8g}; direct local "
            "loss agrees with the earlier algebraic screen. Installed trial "
            f"{c183_hash}: NE pair [48,171], arterial-only common-gate "
            f"adjustment, output multiplier {r178_strength:.8g}; retains "
            "validated guarded insulin correction. Local NLL delta "
            f"{c182_rows[1]['all']:.8g}, held-out "
            f"{c182_rows[1]['validation']:.8g}. This is an UNACCEPTED trial "
            "pending paired GPU factual and dose-rollout checks. "
            "Checked combined fallback retained in c183_fallback.\n")
```