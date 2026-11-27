The recovered evidence shows a substantial gap: the records support roughly −2 glucose units per insulin-rate unit and +3 MAP units per norepinephrine-rate unit, while the original rollout responds weakly. I’ll recover the dynamic estimates and existing repair code next. That should show which candidate can address both drugs without sacrificing factual fit.

```python
import re, types
text115 = Path("/work/notes.md").read_text()
sections115 = re.split(r"(?=^## )", text115, flags=re.M)
print("=== EARLY IDENTIFICATION AND REPAIR SUMMARY ===")
selected115 = [s for s in sections115 if re.match(r"## (Run [6-9]:|Run 1[0-5]:)", s)]
print("\n".join(selected115)[:4200])
print("\n=== NOTEBOOK SECTION TITLES ===")
print("\n".join(s.splitlines()[0] for s in sections115 if s.startswith("## ")))
print("\n=== KERNEL FUNCTIONS ===")
fn115 = sorted(k for k,v in list(globals().items())
               if isinstance(v, types.FunctionType) and v.__module__ == "__main__")
print(" ".join(fn115))
print("\n=== DATAFRAMES ===")
print([(k, v.shape) for k,v in list(globals().items()) if isinstance(v, pd.DataFrame)])
print("\n=== FORWARD FILES ===")
print({p.name: [q.name for q in p.iterdir()] for p in Path("/work").glob("fwd_*")})
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 115: compact recovery\n"
            "- Early cross-fitted estimates support insulin glucose effects near -2 and NE MAP effects near +3 per rate unit.\n"
            "- Recovering dynamic estimates and candidate interfaces; no edits.\n"
            "- Python runs: 115; forward passes: 4.\n")
```