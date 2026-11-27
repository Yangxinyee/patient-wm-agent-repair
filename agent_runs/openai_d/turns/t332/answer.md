The architecture confirms that the edits affect reading-value logits, not the recurrent dynamics. The early record analysis also shows why raw dose–reading associations are unreliable here. I’ll recover the later causal estimates and exact validation cohorts; those—not medical assumptions—will determine whether the stronger norepinephrine trial is worth retaining.

```python
import re

_r277_ids = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"]
)["stay_id"].unique().tolist())
print("=== TUNING COHORTS ===")
print("Total stays:", len(_r277_ids))
for j in range(4):
    cohort = _r277_ids[100*j:100*(j+1)]
    print(j, len(cohort), cohort[0], cohort[-1],
          "consecutive:", cohort == list(range(cohort[0], cohort[-1]+1)))

_r277_text = Path("/work/notes.md").read_text()
print("\n=== LATER CAUSAL ROBUSTNESS CHECKS ===")
_r277_start = _r277_text.find("## Run 4\n")
print(_r277_text[_r277_start:_r277_start+4200])

print("\n=== TARGET-RELATED NOTE EXCERPTS ===")
_r277_lines = _r277_text.splitlines()
_r277_hits = [
    i for i, line in enumerate(_r277_lines)
    if re.search(r"target|g.comput|structural|true.effect|causal.rollout",
                 line, re.I)
]
_r277_output = []
for i in _r277_hits:
    snippet = "\n".join(_r277_lines[max(0, i-1):min(len(_r277_lines), i+3)])
    if snippet not in _r277_output:
        _r277_output.append(snippet)
print("\n---\n".join(_r277_output)[:2600])

print("\n=== FORWARD FILE INTERFACES ===")
for k in (39, 43, 44):
    folder = Path(f"/work/fwd_{k}")
    print(k, [p.name for p in folder.iterdir()])
print("\n=== CHECKPOINT FILE INTERFACE ===")
with np.load("/work/safe_before_ne40.npz") as checkpoint:
    print({k: checkpoint[k].shape for k in checkpoint.files})
```