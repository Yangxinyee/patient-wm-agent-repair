Earlier weight edits produced positive norepinephrine responses, but stronger versions failed factual fit. The restored candidate uses only 20% of that correction. I’ll check the previous compensation attempts and insulin balancing results before testing a safer trade-off; local estimates alone have proved unreliable.

```python
print("=== FRONTIER AND COMPENSATION NOTES ===")
notes267 = Path("/work/notes.md").read_text()
for run in (156, 157, 158, 159, 201, 204, 205, 206, 207,
            225, 226, 229, 230, 231, 232, 233, 235):
    ms = list(re.finditer(rf"(?m)^##[^\n]*\b{run}\b[^\n]*", notes267))
    for m in ms[-1:]:
        end = notes267.find("\n##", m.end())
        print(notes267[m.start():end if end >= 0 else None][:420])

print("\n=== TRAINING-RECORD IDENTIFICATION UPDATES ===")
record_lines267 = [line for line in notes267.splitlines()
    if any(s in line.lower() for s in
           ("beta=", "beta =", "rho=", "rho =", "persistence=", "cluster se",
            "randomized", "instrument", "placebo", "discontinuity"))]
for line in record_lines267[-12:]:
    print(line[:260])

print("\n=== INSULIN ARCHIVES ===")
for fn in ("baseline77.npz", "insulin_candidate63.npz"):
    with np.load("/work/"+fn) as a:
        print(fn, [(k, a[k].shape) for k in a.files])
        print({k: a[k].tolist() for k in a.files if a[k].size <= 8})

print("\n=== FACTUAL SAMPLE IDENTITIES ===")
for k in (2, 9, 21, 24, 26, 28, 31):
    p = Path(f"/work/fwd_{k}/factual_original.csv")
    d = pd.read_csv(p)
    ids = d.stay_id.unique()
    print(k, len(ids), int(ids.min()), int(ids.max()),
          "sequential", bool(np.array_equal(np.sort(ids), np.arange(ids.min(), ids.min()+len(ids)))))

print("\n=== RETAINED RESPONSE/FITTING HELPERS ===")
print([k for k, v in list(globals().items()) if callable(v)
       and any(s in k.lower() for s in
               ("response", "decoder", "local", "predict", "feature", "ridge", "eval", "probe"))][:60])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 267: measured-frontier audit\n"
            "Joint236 restored. Strong NE151 had NE2=1.893 and half=-1.633 "
            "but factual cost=.01107 with full insulin baseline. "
            "Recovering prior compensation outcomes and insulin balancing contracts "
            "before testing a conservative strength reallocation. No weights changed.\n"
            "Python runs: 267; forwards: 32.\n")
```