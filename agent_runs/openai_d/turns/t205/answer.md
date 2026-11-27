The fitted gate cleanly separates cuffs from arterial readings and barely changes under the controlled dose probe. Suppressing cuff effects leaves room for the roughly 1.82× strength suggested by the doubling target, though local held-out NLL is borderline. I’ll install this reversible trial and check the actual GPU factual fit and rollouts before accepting it.

```python
import hashlib

r178_fallback = {n: torch.as_tensor(v).clone()
                 for n, v in r171_combined_backup.items()}
r178_candidate = {n: v.clone() for n, v in r178_fallback.items()}
r178_strength = float(r177_target_multiplier)
r178_rows = list(r170_rows)

for row in r178_rows:
    r178_candidate["attr_heads.value_bin.0.weight"][row] -= torch.tensor(
        r177_gate_strength * r177_v, dtype=torch.float32)
    r178_candidate["attr_heads.value_bin.0.bias"][row] -= float(
        r177_gate_strength * r177_b)
    r178_candidate["attr_heads.value_bin.2.weight"][:, row] *= r178_strength

r178_logits = torch.as_tensor(r145_head(r178_candidate)).numpy()
r178_expected = (r175_logits
                + (r178_strength * r177_new_pair - r175_pair)[..., None]
                * r175_out)
print("Computed-edit logit reconstruction maximum error:",
      float(np.max(np.abs(r178_logits - r178_expected))))

r178_loss = np.asarray(r138_nll(r178_logits))
r178_delta = r178_loss - np.asarray(r170_orig_loss)
print("Local NLL deltas:",
      {"all": float(r178_delta.mean()),
       "fit": float(r178_delta[r177_fit].mean()),
       "validation": float(r178_delta[r177_val].mean())})

assert np.isfinite(r178_logits).all()
assert np.max(np.abs(r178_logits - r178_expected)) < 0.002
for n, value in r178_candidate.items():
    assert tuple(value.shape) == tuple(SHAPES[n])
    save_edit(n, value)

r178_hash = hashlib.sha256(b"".join(
    r178_candidate[n].numpy().tobytes() for n in sorted(r178_candidate)
)).hexdigest()[:12]
print("Installed unaccepted selective-NE trial:", r178_hash, edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 174–178: selective NE trial\n"
            "Forward 22 versus insulin-only Forward 20 isolates NE factual penalty: "
            "glucose negligible, MAP arterial -0.000108, MAP cuff +0.006995. "
            "Blind NE amplification raises local loss. Fitted ridge cuff classifier "
            "on 75 factual-state stays, held out 25, constrained orthogonal to mean "
            "controlled NE probe. Held-out scores separate cuff (~1) from other codes "
            "(~0), and probe score changes are tiny. Subtracted 80*classifier from "
            "both NE neurons' common gate, retaining dose direction. Scaled NE "
            f"output by {r178_strength:.6f}, derived from record doubling target "
            "relative to measured original/current rollout effects. "
            f"Candidate {r178_hash}; cached NLL delta "
            f"{r178_delta.mean():.6f}, held-out {r178_delta[r177_val].mean():.6f}. "
            "Trial is unaccepted pending GPU factual and dose checks. Prior passing "
            "combined edit retained in r178_fallback/r171_combined_backup; "
            "guarded insulin unchanged.\n")
```
