The recovered notes show substantial dose-selection confounding and record-based effect estimates, not hidden truth. The output was truncated before the final cohort and weight checks. I’ll retrieve those compactly: matching weights and cohort will justify the last paired rollout; a mismatch would invalidate that comparison.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
import json
import re

final_rollout_stays = sorted(
    int(x) for x in pd.read_csv(
        "/work/fwd_48/factual_edited.csv"
    )["stay_id"].unique()
)
assert len(final_rollout_stays) == 100
print("FINAL_FORWARD_STAYS:", json.dumps(final_rollout_stays))

current_weight = load("core.b_in.weight")
if hasattr(current_weight, "detach"):
    current_weight = current_weight.detach().cpu().numpy()
current_weight = np.ascontiguousarray(current_weight, dtype=np.float32)
current_sha = hashlib.sha256(current_weight.tobytes()).hexdigest()
print("CURRENT_SHA256:", current_sha)
print("EXPECTED_SHA256:",
      "c93a8cf4fad68f4cec97c3f7c5cc1c2ceea479611ba573bd115eb6df4f99bbc2")
assert current_sha == (
    "c93a8cf4fad68f4cec97c3f7c5cc1c2ceea479611ba573bd115eb6df4f99bbc2"
)
assert set(edited()) == {"core.b_in.weight"}
print("EDIT_VERIFIED: only core.b_in.weight, candidate C unchanged")

for filename in ["joint274.npz", "joint284.npz", "joint291.npz"]:
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        compact = {}
        for key in archive.files:
            value = archive[key]
            compact[key] = (
                value.tolist() if value.size <= 8
                else {"shape": list(value.shape)}
            )
        print("ARCHIVE", filename, compact)

notes_text = Path("/work/notes.md").read_text()
pattern = re.compile(
    r"beta\s*[=:]|persistence\s*[=:]|impulse targets|"
    r"NE DML|identified effects|statistical targets",
    re.I,
)
evidence_lines = [
    line for line in notes_text.splitlines() if pattern.search(line)
]
print("RECORD-IDENTIFICATION SUMMARY:")
for line in evidence_lines[-10:]:
    print(line[:650])

with Path("/work/notes.md").open("a") as handle:
    handle.write(
        "\n## Run 295: compact final preflight\n"
        "Verified unchanged candidate C by SHA256 and sole edited tensor. "
        "Recovered exact forward48 rollout IDs and compact record-effect notes. "
        "No weights changed. Final paired dose rollout is next. "
        "Python runs: 295; forwards: 49.\n"
    )
```