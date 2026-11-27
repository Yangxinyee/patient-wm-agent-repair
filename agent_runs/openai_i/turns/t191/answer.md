The retained posterior edit is reversible, but its locally estimated response directions have failed the rollout test. I’ll try a bounded sign-and-gain probe to measure their actual influence. Stronger negative insulin effects and positive norepinephrine effects would support further calibration; saturation or another wrong-way response would rule out this pathway.

```python
import numpy as np
from pathlib import Path

print("=== RECORD-BASED EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:6500])

print("\n=== LATENT DIRECTION SCALES ===")
for drug, direction in latent_directions161.items():
    d = direction.detach().cpu().numpy() if hasattr(direction, "detach") else np.asarray(direction)
    print(drug, "norm", float(np.linalg.norm(d)), "rms", float(np.sqrt(np.mean(d*d))))
print("Factual feature RMS:", np.sqrt(np.mean(Fbase169**2, axis=0)))

install169(-40.0, -20.0)
print("\nActive edits:", edited())
for name in edited():
    difference = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, "change RMS", float(np.sqrt(np.mean(difference**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 172 — empirical posterior sign/gain calibration\n"
            "- Installed response-based posterior probe with gains insulin -40, NE -20.\n"
            "- This is a reversible pathway test, not a causal-effect estimate or accepted repair.\n"
            "- Requires matched rollout and factual validation; reject if influence saturates or remains wrong-way.\n")
```