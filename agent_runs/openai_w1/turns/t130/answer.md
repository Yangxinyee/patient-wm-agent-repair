Recomputation still matches drug bundles but not reading bundles, so stale inputs were not the explanation. I’ll test normalized-empty versus zero action encoding directly. The earlier value-head innovation decoder also failed held-out validation; I’ll instead investigate a gated posterior edit that acts only when a drug is recorded.

```python
import numpy as np, torch, re
from pathlib import Path

rr119_u_nonzero = rr_encode_atoms112(rr_atoms4, rr_act_mask112, "encode_action", False)
print("=== EMPTY-ACTION TEST ===")
for label, u in (("zero", rr117_u), ("normalized empty", rr119_u_nonzero)):
    z = rr_mlp(torch.cat((rr_best_ctx111, rr117_e, u), 1), "z_post", "gelu")[:, :32]
    err = (z - rr_zexport4).square().mean(1).sqrt().numpy()
    for subset, mask in (
        ("reading", np.asarray(rr_has_reading4, bool)),
        ("valid other", np.asarray(rr_valid4, bool) &
         ~np.asarray(rr_has_reading4, bool) & ~np.asarray(rr_has_active4, bool))
    ):
        if mask.any():
            print(label, subset, np.quantile(err[mask], [0, .5, .9, 1]))
print("cached local reading action-vector norms:",
      np.quantile(r86_u.norm(dim=1).numpy(), [0, .5, 1]))
print("empty action-vector norm:",
      float(rr119_u_nonzero[torch.as_tensor(~np.asarray(rr_has_active4, bool))][0].norm()))

print("\n=== POSTERIOR-CANDIDATE DETAILS ===")
for key, value in rr51_assignment.items():
    print("assignment", key, type(value).__name__)
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            print(i, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:120])
    elif isinstance(value, dict):
        print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
               for k, v in value.items()])
for key, value in rr51_trace_candidates.items():
    print("trace candidate", key, type(value).__name__)
    if isinstance(value, (tuple, list)):
        print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:80])
               for v in value])
    elif isinstance(value, dict):
        print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
               for k, v in value.items()])

print("\n=== REMAINING POSTERIOR NOTES ===")
sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for s in sections:
    title = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"\bRun (5[2-9]|6[0-5])\b", title):
        print(s[:650])

print("\n=== FRESH BUNDLE METADATA OBJECTS ===")
for n, v in sorted(list(globals().items())):
    if n.startswith("rr_") and isinstance(v, (list, tuple)):
        print(n, type(v).__name__, len(v), repr(v[:1])[:200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 119\nFresh posterior remains exact on drug bundles but not reading bundles. "
            "Directly testing empty-action handling. Rejecting the prior value-head innovation decoder "
            "because held-out R2 was negative; considering a gated action/posterior repair instead. "
            "No edits. Counts: 119 Python runs, 6 forward passes.\n")
```
