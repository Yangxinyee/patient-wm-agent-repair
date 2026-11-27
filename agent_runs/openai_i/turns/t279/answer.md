Sharpening the selector did not help: its factual loss is still +0.10705 nats, so that route is rejected. The records support insulin lowering glucose and norepinephrine raising MAP, but the weight edit has not preserved factual predictions. I’ll inspect the replay helpers and earlier candidates to find a less disruptive route. It must reproduce baseline states and carry a dose response without shifting untreated predictions.

```python
print("=== REPLAY FUNCTIONS ===")
funcs248 = {}
for name, value in list(globals().items()):
    if inspect.isfunction(value) and value.__module__ == "__main__":
        funcs248[name] = value
print("names:", " ".join(funcs248))
for name, value in funcs248.items():
    if any(t in name.lower() for t in ("prelude", "replay", "readout", "local", "step", "route", "head")):
        print(name, str(inspect.signature(value)))

print("\n=== IMPORTANT OBJECTS ===")
for name, value in list(globals().items()):
    if any(t in name.lower() for t in ("tune", "causal", "direction", "slope", "candidate", "base222", "route", "base209", "neutral")):
        if isinstance(value, np.ndarray):
            print(name, value.shape)
        elif isinstance(value, dict):
            print(name, "dict", len(value), list(value)[:10])
        elif isinstance(value, (list, tuple)) and len(value) < 410:
            print(name, type(value).__name__, len(value), str(list(value)[:5])[:100])
        elif isinstance(value, (int, float, str)):
            print(name, str(value)[:100])

print("\n=== RELEVANT HISTORY ===")
for s in sections246:
    h = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"\b(?:18[6-9]|19[0-9]|20[0-9]|21[0-6])\b", h):
        body = "\n".join(s.splitlines()[:5])
        print(body[:470])

print("\n=== PRIOR FORWARD SUMMARIES ===")
for k in range(10, 20):
    p = Path(f"/work/fwd_{k}/summary.json")
    if not p.exists():
        continue
    j = json.loads(p.read_text())
    print(k, j.get("type"), j.get("n_stays"),
          {m: {key: val for key, val in r.items()
                if isinstance(val, (int, float, str, list)) and key != "file"}
           for m, r in j.get("results", {}).items()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 248 — reject sharp243; recover simpler pathways\n"
            "- Forward22 sharp243 factual dNLL +.10705: selector sharpening did not rescue factual fit.\n"
            "- Neutral209 remains installed. No attention-response candidate is accepted.\n"
            "- Recovering retained replay functions and earlier latent/head validation before a simpler edit.\n")
```