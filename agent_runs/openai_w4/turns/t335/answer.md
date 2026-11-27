The architecture confirms that dose information enters the recurrent update through the action embedding and latent posterior. The recovery output was truncated before showing the validation IDs. I’ll retrieve just those IDs and the latest result schemas so the next GPU checks are directly comparable.

```python
from pathlib import Path
import pandas as pd

dirs288 = sorted(Path("/work").glob("fwd_*"),
                 key=lambda p: int(p.name.split("_")[-1]))
facts288 = [p for d in dirs288 for p in d.glob("factual_edited.csv")]
latest_fact288 = facts288[-1]
fact288 = pd.read_csv(latest_fact288)
idcol288 = next(c for c in ("stay_id", "stay", "subject_id")
               if c in fact288.columns)
validation_ids288 = fact288[idcol288].drop_duplicates().tolist()
print("FACTUAL FILE", str(latest_fact288))
print("FACTUAL COLUMNS", fact288.columns.tolist())
print("VALIDATION STAYS", len(validation_ids288), validation_ids288)

dosepaths288 = [p for d in dirs288 for p in d.glob("dose_edit_edited.csv")]
if dosepaths288:
    dose288 = pd.read_csv(dosepaths288[-1])
    print("DOSE FILE", str(dosepaths288[-1]))
    print("DOSE COLUMNS", dose288.columns.tolist())
    print(dose288.head(3).to_string(index=False))

notes288 = Path("/work/notes.md").read_text()
print("INITIAL IDENTIFICATION NOTES")
print(notes288[:2200])
print("CANDIDATE", chosen286)
print("LATEST DIRECTORY", dirs288[-1].name)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 288\nRecovered exact GPU validation IDs and output "
            "schemas after truncated recovery. No weights changed. "
            "Run-286 candidate awaits paired GPU validation.\n")
```