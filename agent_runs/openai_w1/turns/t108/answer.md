The latest result confirms that no weights are edited, but the retained encoder helper has obsolete unit IDs. I’ll reread the architecture before rebuilding the calibration from consistent exports. A reconstructed encoder must match the exported states before I trust its proposed edit directions.

```python
from pathlib import Path

rr_architecture = Path("/model/ARCHITECTURE.md").read_text()
print(rr_architecture)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 98\n"
        "No weights edited. Retained r33 atom helper has obsolete unit IDs; "
        "do not reuse it uncorrected. Reading architecture before rebuilding "
        "a coherent calibration from original state exports.\n"
        "Counts: 98 Python runs, 6 forward passes.\n"
    )
```
