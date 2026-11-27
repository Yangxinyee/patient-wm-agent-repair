The GPU penalty is still +0.00626 nats on the first cohort, so this candidate is not yet safe to submit. The correction helped only slightly there. I’ll inspect where the loss occurs and recover the broader validation caches before deciding whether to reduce the edit. A pooled result with margin below +0.005—not this cohort alone—would support keeping it.

```python
r258_factual = {
    m: pd.read_csv(f"/work/fwd_36/factual_{m}.csv")
    for m in ("original", "edited")
}
print("=== FORWARD 36 FILE SCHEMA ===")
print(r258_factual["original"].columns.tolist())
r258_nll_cols = [
    c for c in r258_factual["original"].columns if "nll" in c.lower()
]
assert len(r258_nll_cols) == 1
r258_nll_col = r258_nll_cols[0]
r258_delta = (
    r258_factual["edited"][r258_nll_col].to_numpy()
    - r258_factual["original"][r258_nll_col].to_numpy()
)
print("Exact overall delta:", float(r258_delta.mean()))
for c in ("event", "event_name", "code_name"):
    if c in r258_factual["original"]:
        print(pd.DataFrame({
            "event": r258_factual["original"][c], "delta": r258_delta
        }).groupby("event").delta.agg(["count", "mean"]).to_string())
        break

print("\n=== EXISTING STATE FILES ===")
for p in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    print(str(p), "MB=", round(p.stat().st_size / 1e6, 2))

print("\n=== CACHE BUILDING FUNCTIONS ===")
r258_g = dict(globals())
r258_functions = [
    (n, v) for n, v in r258_g.items()
    if callable(v) and hasattr(v, "__code__")
    and (n.startswith(("r210_", "r211_", "r235_"))
         or re.search(r"cache|with_code|head_features|build.*feature", n))
]
for n, v in r258_functions[:30]:
    try:
        print(n, inspect.signature(v), "uses:", v.__code__.co_names)
    except Exception:
        print(n)

print("\n=== EXISTING READING CACHES ===")
for n, v in sorted(r258_g.items()):
    if isinstance(v, dict) and "y" in v and (
        "event" in v or "X" in v or "x" in v
    ):
        print(n, "n=", len(v["y"]), "keys=", list(v.keys())[:16])

print("\n=== EARLY ROLLOUT TARGETS ===")
print(r255_notes[10300:13900])

print("\n=== SAFE CHECKPOINT NOTES ===")
for s in r255_sections:
    if re.search(r"Run(?:s)? (?:23[45]|210)", s.split("\n", 1)[0]):
        print(s[:1500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 36 / Run 258: first-cohort cancellation remains insufficient\n"
            "Shrink80-scale4 previous-bin cancellation gave first100 tuning "
            f"factual delta {r258_delta.mean():.8f}, still above .005. "
            "Candidate retained only for diagnosis, not accepted. Inspecting "
            "available caches and safe fallback before broader validation or reduction.\n")
```