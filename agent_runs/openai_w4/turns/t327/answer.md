The architecture confirms that dose edits enter through the action encoder and recurrent state. The recovered record analysis supports insulin lowering glucose and norepinephrine raising MAP, but the output was truncated before the rollout targets and validation IDs. I’ll retrieve those details; they will distinguish a successful calibration from merely producing effects with the right signs.

```python
print("RECORD-BASED TARGET NOTES")
budget282 = 4600
for s in sections281:
    if re.search(r"rollout target|target shift|true shift|cumulative|effect estimate|impulse|dynamics|identification|estimated.*shift", s, re.I):
        text = s[:1100]
        if len(text) <= budget282:
            print(text)
            budget282 -= len(text)
        if budget282 < 500:
            break

print("\nLATEST VALIDATION FILES")
for k in (29, 30, 31, 32):
    p = Path(f"/work/fwd_{k}")
    if not p.exists():
        continue
    print(k, [q.name for q in sorted(p.iterdir())])
    for q in sorted(p.glob("*.csv")):
        df = pd.read_csv(q)
        print(q.name, df.shape, df.columns.tolist())
        idcols = [c for c in df.columns if "stay" in c.lower() and "hour" not in c.lower()]
        if idcols:
            validation_ids282 = [int(v) for v in df[idcols[0]].drop_duplicates()]
            print("ID count/range", len(validation_ids282), min(validation_ids282), max(validation_ids282))
print("VALIDATION IDS", validation_ids282)

print("\nHELPERS")
helper_names282 = [
    k for k in list(globals())
    if callable(globals()[k]) and re.search(r"candidate|sym|asym|gate|factual_eval194|atom_clean58|rp_linear|rp_mlp", k)
]
for k in helper_names282:
    try:
        print(k, inspect.signature(globals()[k]))
    except Exception:
        pass

print("\nLATEST NOTE HEADINGS")
print("\n".join(s.splitlines()[0] for s in sections281[-35:] if s.strip()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 282\nBounded recovery of target-estimation notes, validation stay IDs and calibration helper names after Run-281 output truncation. No weights changed.\n")
```
