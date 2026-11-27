The provisional rollout targets are insulin −13.60/+6.80 and norepinephrine +4.75/−2.37. The guarded strength-2.6 insulin candidate has lower cached loss than the rejected unguarded trial. I’ll install it temporarily; passing the previously problematic factual cohort would support the guard hypothesis.

```python
import hashlib

r164_fallback = {
    name: np.array(r157_np(value), dtype=np.float32, copy=True)
    for name, value in r115_strength2.items()
}
r164_candidate = {name: value.copy() for name, value in r164_fallback.items()}
r164_guard = r159_candidates[("warmup", 4, 1.0)]
r164_gv = np.asarray(r164_guard["v"], dtype=np.float64)
r164_gb = float(r164_guard["b"])
r164_dv = np.asarray(r157_decoder, dtype=np.float64)
r164_db = float(r157_decoder_b)

r164_candidate["attr_heads.value_bin.0.weight"][r159_p] = r164_gv + r164_dv
r164_candidate["attr_heads.value_bin.0.weight"][r159_m] = r164_gv - r164_dv
r164_candidate["attr_heads.value_bin.0.bias"][r159_p] = r164_gb + r164_db
r164_candidate["attr_heads.value_bin.0.bias"][r159_m] = r164_gb - r164_db
r164_candidate["attr_heads.value_bin.2.weight"][:, r159_p] = 1.3 * r159_C
r164_candidate["attr_heads.value_bin.2.weight"][:, r159_m] = -1.3 * r159_C

r164_G = r159_X @ r164_gv + r164_gb
r164_D = r159_X @ r164_dv + r164_db
r164_logits = r159_const + (
    1.3 * r159_pair(r164_G, r164_D)[..., None] * r159_C
)
r164_loss = r138_nll(r164_logits)
r164_delta = float(np.mean(r164_loss - r159_original_loss))
print("cached factual NLL delta:", r164_delta)
assert abs(r164_delta - 0.002025) < 0.0001

for name, value in r164_candidate.items():
    save_edit(name, value)
r164_hash = hashlib.sha256(
    b"".join(r164_candidate[name].tobytes() for name in sorted(r164_candidate))
).hexdigest()[:12]
print("installed guarded trial:", r164_hash)
print("edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        f"\n## Run 164: guarded insulin trial\n"
        f"Installed strength-2.6 insulin with warmup support guard "
        f"(penalty 4, threshold 1), hash {r164_hash}. "
        f"Cached factual NLL delta {r164_delta:.6f}; cached held-out "
        f"delta approximately +0.003182. Strength-2 fallback retained "
        f"in r164_fallback and r115_strength2. This is an unaccepted "
        f"trial pending GPU factual and dose-rollout checks.\n"
    )
```
