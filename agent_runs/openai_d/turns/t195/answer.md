The previous NE correction improved rollout direction but was calibrated using a diagnostic probe, not the evaluation intervention. The guarded insulin edit has freed some factual-loss margin. I’ll combine it with that NE correction unchanged; a paired factual check will determine whether this is a viable combined fallback.

```python
import numpy as np, torch, hashlib
from pathlib import Path

r170_names = [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias",
]
r170_insulin_backup = {
    n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in r170_names
}
r170_ne_patch = {
    n: torch.as_tensor(r139_candidate[n], dtype=torch.float32)
       - torch.as_tensor(r115_strength2[n], dtype=torch.float32)
    for n in r170_names
}
r170_rows = torch.where(
    r170_ne_patch["attr_heads.value_bin.0.weight"].abs().max(dim=1).values > 1e-6
)[0].tolist()
print("NE input rows changed:", r170_rows)
assert set(r170_rows).issubset(set(np.asarray(r138_slots).tolist()))

r170_candidate = {
    n: r170_insulin_backup[n] + r170_ne_patch[n] for n in r170_names
}
r170_local_logits = r145_head(r170_candidate) + r145_transition
r170_local_loss = r138_nll(r170_local_logits.detach().cpu().numpy())
r170_orig_loss = np.asarray(r145_loss_orig)
r170_delta = r170_local_loss - r170_orig_loss
print("Cached factual NLL delta:", float(r170_delta.mean()))
print("Cached fit / held-out deltas:",
      float(r170_delta[r137_fit].mean()), float(r170_delta[r137_val].mean()))
for r170_event in sorted(head_meta.event.unique()):
    r170_mask = head_meta.event.to_numpy() == r170_event
    print(r170_event, int(r170_mask.sum()), float(r170_delta[r170_mask].mean()))

for n, t in r170_candidate.items():
    save_edit(n, t)
r170_hash = hashlib.sha256(
    b"".join(r170_candidate[n].numpy().tobytes() for n in sorted(r170_names))
).hexdigest()[:12]
print("Installed reversible combined trial:", r170_hash)
with Path("/work/notes.md").open("a") as f:
    f.write(
        f"\n## Run 170: combined guarded-insulin / prior-NE trial\n"
        f"Installed four-tensor candidate {r170_hash}: retained validated guarded "
        f"strength-2.6 insulin and added the prior NE correction unchanged. "
        f"Cached overall NLL delta {r170_delta.mean():.6f}, held-out "
        f"{r170_delta[r137_val].mean():.6f}. These are diagnostic only; "
        f"candidate is unaccepted pending GPU factual and actual rollout checks. "
        f"Guarded insulin fallback is preserved in r170_insulin_backup.\n"
    )
```
