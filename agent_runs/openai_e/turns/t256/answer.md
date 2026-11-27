The trial passes the factual check, but narrowly: NLL increased by 0.00468, leaving little room for another edit. I’ll review the measured trade-offs and record-derived targets before choosing a next step. I’m looking for a correction that improves the dose responses without exceeding the factual limit.

```python
notes221 = Path("/work/notes.md").read_text()
print("=== EARLY EVIDENCE ===")
print(notes221[:2600])

print("\n=== NOTEBOOK SECTION INDEX ===")
headers221 = [(m.start(), m.group(0)) for m in re.finditer(r"(?m)^## .*", notes221)]
for pos, title in headers221:
    if any(w in title.lower() for w in (
        "causal", "identif", "calibr", "direction", "frontier", "transition",
        "run 63:", "run 76:", "run 77:", "run 82:", "run 107:", "run 155:",
        "run 162:", "run 175:", "run 214:"
    )):
        print(pos, title)

print("\n=== GPU VALIDATION SUMMARY ===")
for i in range(1, 26):
    p = Path(f"/work/fwd_{i}/summary.json")
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    typ = s.get("type")
    rs = s.get("results", {})
    if typ == "factual":
        vals = {k: round(v.get("value_nll_per_value", np.nan), 6) for k, v in rs.items()}
        print(i, typ, s.get("split"), s.get("n_stays"), vals, s.get("edit_hash"))
    elif typ == "dose_edit":
        vals = {k: {e: v.get("mean_shift") for e, v in r.get("edits", {}).items()}
                for k, r in rs.items()}
        print(i, typ, s.get("n_stays"), vals)

print("\n=== TARGETED PERSISTENT OBJECTS ===")
selected221 = [k for k in list(globals()) if re.search(
    r"(?:fit_impulse76|record_schedule_target69|decode|static|calib|probe|safety|guard|train_meta162|pt_.*175|.*214|.*219)$", k
)]
for k in selected221[:65]:
    v = globals()[k]
    if callable(v):
        import inspect
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, "function", sig)
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:8])
    elif isinstance(v, (float, int, str)):
        print(k, repr(v)[:100])
    else:
        print(k, type(v).__name__)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 221: trial219 passes narrowly\n"
            "Forward24 paired factual NLL: original 1.63461499, edited 1.63929569, cost .00468070. "
            "Trial219 is admissible on this sample but has only .00031930 margin. "
            "Recovering measured frontier, causal targets, and calibration history before further edits. "
            "Python runs: 221; forwards: 25.\n")
```