The stronger norepinephrine candidate produced only a small x2 gain, weakened the half-dose response, and worsened factual fit. I won’t retain it. I’ll test a modest blend of the validated fallback with the earlier joint-dose candidate, which improved all four local contrasts. Better rollouts within the factual limit would support the blend; otherwise I’ll restore the fallback.

```python
from pathlib import Path
import re
import numpy as np
import torch

# First restore the fully audited fallback.
for n, t in backup296.items():
    save_edit(n, torch.as_tensor(t).float())

head298_names = [
    "attr_heads.value_bin.2.weight",
    "attr_heads.value_bin.2.bias",
]
blend298_alpha = 0.20
blend298_staged = False

if isinstance(globals().get("backup292_strong"), dict):
    strong298 = backup292_strong
    assert all(n in strong298 for n in backup296)
    for n in backup296:
        if n not in head298_names:
            err = float(
                (torch.as_tensor(strong298[n]) -
                 torch.as_tensor(backup296[n])).abs().max()
            )
            assert err < 1e-6, (n, err)

    for n in head298_names:
        base = torch.as_tensor(backup296[n]).float()
        strong = torch.as_tensor(strong298[n]).float()
        save_edit(n, (1-blend298_alpha)*base + blend298_alpha*strong)
    blend298_staged = True
    print("STAGED: 80% audited fallback + 20% joint-dose strong final head")
else:
    print("Strong backup unavailable; audited fallback remains restored.")

backup298_candidate = {
    n: torch.as_tensor(load(n)).detach().clone() for n in edited()
}
for n in head298_names:
    print(n, "max change from fallback:",
          float((backup298_candidate[n] -
                 torch.as_tensor(backup296[n])).abs().max()))

# Bounded recovery of the records-based evidence for the final explanation.
print("=== EARLY CAUSAL-EVIDENCE NOTE EXCERPTS ===")
notes298 = Path("/work/notes.md").read_text()
for pattern in [
    r"(?mi)^#{1,3}[^\n]*\brun\s*23\b[^\n]*",
    r"(?mi)^#{1,3}[^\n]*(?:DML|threshold|quiet.channel)[^\n]*",
]:
    matches = list(re.finditer(pattern, notes298))
    for m in matches[:3]:
        print(notes298[m.start():m.start()+1300])

print("=== SMALL TARGET / EFFECT OBJECTS ===")
for n in list(globals()):
    if n.startswith("_") or not re.search(
        r"(dml|kernel23|target23|effect23|coef23)", n, re.I
    ):
        continue
    obj = globals()[n]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(n, tuple(obj.shape),
              obj if obj.ndim <= 1 and obj.numel() <= 24
              else "" if isinstance(obj, torch.Tensor)
              else obj if obj.ndim <= 1 and obj.size <= 24 else "")
    elif isinstance(obj, dict):
        print(n, "dict keys:", list(obj)[:12])
    elif isinstance(obj, (int, float, str)):
        print(n, str(obj)[:200])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 298 / reject NE-focused head; conservative joint-head test\n"
        "Forward35, NE-focused hash a1da07951e: factual delta +.00515 on "
        "tuning100–199. Forward36: shifts [-10.002,+5.013,+2.482,-1.111], "
        "versus audited fallback [-10.201,+4.921,+2.423,-1.151]. "
        "The small NE-x2 gain does not justify worse factual fit and weaker "
        "NE-half. Restored backup296, then staged a final-head-only "
        f"80/20 fallback/backup292_strong blend: {blend298_staged}. "
        "Other five edited tensors are unchanged. GPU factual and rollout "
        "audits are required before accepting this blend.\n"
    )
```
