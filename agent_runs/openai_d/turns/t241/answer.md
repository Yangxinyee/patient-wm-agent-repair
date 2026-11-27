The working kernel contains a fresh tuning cache and backups for insulin-only, weaker combined, and stronger combined edits. The inventory error did not change any weights. I’ll compare these candidates on identical cached inputs; if insulin alone accounts for most of the factual loss, revising norepinephrine alone will not solve the problem.

```python
import inspect
print("=== EVALUATOR ===")
print("signature:", inspect.signature(r201_eval))
print("referenced names:", r201_eval.__code__.co_names)
print("metadata columns:", list(r201_meta.columns))
print("comparison columns:", list(r201_check.columns))

c206_candidates = {
    "original": {n: load_original(n) for n in r170_names},
    "insulin_only": r165_insulin_backup,
    "combined_weak": c183_fallback,
    "combined_strong": c185_current,
}
c206_results = {}
for label, candidate in c206_candidates.items():
    ans = r201_eval(candidate)
    c206_results[label] = ans
    if isinstance(ans, (tuple, list)):
        print(label, "outputs:",
              [(type(a).__name__, getattr(a, "shape", None)) for a in ans])
    elif isinstance(ans, dict):
        print(label, "outputs:", list(ans))
    else:
        print(label, type(ans).__name__, getattr(ans, "shape", None))

print("=== EDITED HIDDEN ROWS / OUTPUT COLUMNS ===")
for label, candidate in c206_candidates.items():
    if label == "original":
        continue
    w1 = np.asarray(candidate["attr_heads.value_bin.0.weight"])
    w2 = np.asarray(candidate["attr_heads.value_bin.2.weight"])
    o1 = np.asarray(load_original("attr_heads.value_bin.0.weight"))
    o2 = np.asarray(load_original("attr_heads.value_bin.2.weight"))
    print(label,
          "input rows", np.flatnonzero(np.max(np.abs(w1-o1), axis=1) > 1e-6).tolist(),
          "output columns", np.flatnonzero(np.max(np.abs(w2-o2), axis=0) > 1e-6).tolist())

print("=== CAUSAL EVIDENCE, RUN 4 ===")
a = _notes.find("## Run 4")
b = _notes.find("\n## ", a+4)
print(_notes[a:min(b if b > a else a+2800, a+2800)])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 206: candidate decomposition audit\n"
            "Recovered the independent tuning 100–199 head cache and evaluated original, "
            "guarded insulin-only, weaker combined, and stronger combined backups on identical "
            "inputs. No weights changed; output interpretation follows in the next run.\n")
```
