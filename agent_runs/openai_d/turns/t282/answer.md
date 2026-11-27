Modest cancellation does buy some margin: NE amplitude 0.50 has nearly the same cached factual cost as the validated 0.40 edit. Its leakage into other reading types remains a concern. I’ll install this candidate after an exact cache check, then test the real rollout. It is worthwhile only if NE improves without materially weakening insulin or violating factual fit.

```python
import hashlib, numpy as np

r242_reg = r240_regressions[(16, 0.001)]
r242_factor = 0.50/0.40
r242_trial = {k: v.copy() for k, v in r239_safe.items()}
r242_trial[r239_keys[2]][:, 48] *= r242_factor
r242_trial[r239_keys[2]][:, 171] *= r242_factor
r242_trial[r239_keys[2]] -= (
    r242_factor * np.outer(r240_q, r242_reg["beta"])
).astype(np.float32)
r242_trial[r239_keys[3]] -= (
    r242_factor * r240_q * r242_reg["intercept"]
).astype(np.float32)

print("=== EXACT CANDIDATE VERIFICATION ===")
r242_verified = {}
for label, c in (("train", c210_train), ("tune", c210_tune)):
    loss, mean, probs = c210_eval(r242_trial, c, True)
    scalar = ((r242_factor-1)*r240_pre[label]["R"]
              - r242_factor*r242_reg["pred"][label])
    fast_loss = r240_loss(c, r240_pre[label], scalar)
    err = float(np.max(np.abs(loss-fast_loss)))
    assert err < 2e-5, (label, err)
    r242_verified[label] = loss-c["loss0"]
    print(label, "NLL delta", float(np.mean(r242_verified[label])),
          "fast/exact max difference", err)
print("train fit", float(r242_verified["train"][r240_fit].mean()),
      "train validation", float(r242_verified["train"][r240_val].mean()))

r242_hash = hashlib.sha256(b"".join(
    k.encode()+r242_trial[k].tobytes() for k in sorted(r242_trial)
)).hexdigest()[:10]
np.savez("/work/ne50_hidden_cancellation.npz", **r242_trial)
for k, v in r242_trial.items():
    save_edit(k, v)
print("Installed candidate:", r242_hash, "edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 242: install modest nonlinear-cancellation trial\n"
            f"Installed candidate {r242_hash}: original safe insulin unchanged, NE "
            "pair output amplitude increased .40 -> .50, with training-only ridge "
            "cancellation through 188 retained hidden features and output bias. "
            "Protected approximate NE probe event means and top 16 derivative PCs. "
            "Exact evaluator matches fast screen. Cached tune100 delta "
            f"{np.mean(r242_verified['tune']):.7f}. "
            "Actual dose-edit response and GPU factual fit unverified. Exact safe "
            "fallback remains r235_safe and /work/safe_6288c226d6.npz.\n")
```