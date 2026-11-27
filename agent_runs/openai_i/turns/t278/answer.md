The installed model has only the six latent-channel edits, not the rejected response route. The visible factual tests failed: +0.00509 nats for the neutral route and +0.10705 for the response route. I’ll check the sharper candidate’s result and recover the record-derived effect estimates. Sharper gating is useful only if it removes factual damage while preserving dose sensitivity.

```python
import inspect

print("=== FORWARD 22 SUMMARY ===")
p247 = Path("/work/fwd_22/summary.json")
print(p247.read_text() if p247.exists() else "No summary")
p247 = Path("/work/fwd_22")
if p247.exists():
    print("files", [f.name for f in p247.iterdir()])

print("\n=== CANDIDATES VERSUS INSTALLED ===")
for fn in ("neutral209.npz", "fallback188.npz", "response233.npz", "sharp243.npz"):
    with np.load("/work/"+fn, allow_pickle=False) as z:
        common = [k for k in z.files if k in NAMES]
        print(fn, "keys", z.files)
        print("tensor differences", {
            k: round(float(np.linalg.norm(z[k]-np.asarray(load(k)))), 6)
            for k in common
        })

print("\n=== RECORD-EVIDENCE NOTE EXCERPTS ===")
for s in sections246:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if any(re.search(rf"\b{r}\b", heading) for r in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,186,188,203,204,209)):
        print(s[:1050])

print("\n=== REPLAY HELPER SIGNATURES ===")
for name, value in list(globals().items()):
    if inspect.isfunction(value) and value.__module__ == "__main__":
        try:
            sig = str(inspect.signature(value))
        except Exception:
            sig = "?"
        print(name, sig)
print("\n=== SELECTED RETAINED OBJECTS ===")
for name, value in list(globals().items()):
    if any(t in name.lower() for t in ("tune", "causal", "sharp", "direction", "slope", "candidate", "base222", "orig", "model", "route")):
        if isinstance(value, np.ndarray):
            print(name, "array", value.shape)
        elif isinstance(value, (list, tuple, dict)):
            print(name, type(value).__name__, len(value), str(list(value)[:6])[:220])
        elif isinstance(value, (int, float, str)):
            print(name, str(value)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 247 — compact candidate audit\n"
            "- Actual installed edit is the six-tensor latent candidate, not the rejected attention response route.\n"
            "- Reconfirmed small-route factual dNLL +.005087 and response233 +.107049 on first 100 tuning stays.\n"
            "- Checking forward22 and recovering record evidence and retained replay contracts; no weights changed.\n")
```