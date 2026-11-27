The screen failed because the replay helper requires all 181 tensors, while the backup contains only the ten edited ones. The `finally` block restored the weight dictionaries, and saved weights were unchanged. I’ll supply complete dictionaries and rerun the control and candidate screens; matching the earlier control NLL will confirm the setup is correct.

```python
def full273(candidate):
    return {
        n: torch.as_tensor(
            a262(candidate[n] if n in candidate else load(n)),
            dtype=torch.float32
        ).clone()
        for n in NAMES
    }

r273_snapshots = [
    (name, obj, dict(obj))
    for name, obj in list(globals().items())
    if isinstance(obj, dict) and "z_post.0.weight" in obj
]
r273_results, r273_candidates = {}, {}
try:
    r273_control = full273(backup269)
    _, losses, _ = factual_eval194(r273_control)
    r273_control_nll = float(np.mean(losses))
    print("CURRENT CPU CONTROL", r273_control_nll,
          "expected", 1.6392319202423096,
          "tensor count", len(r273_control))
    for g, exact in [(0.8, False), (1.0, False), (0.8, True), (1.0, True)]:
        c = full273(symmetric_ne272(g, exact))
        _, losses, _ = factual_eval194(c)
        nll = float(np.mean(losses))
        r273_results[(g, exact)] = {
            "nll": nll, "delta": nll - 1.6355364322662354
        }
        r273_candidates[(g, exact)] = c
    print("UNSAVED NE SCREEN", r273_results)
finally:
    for name, obj, snapshot in r273_snapshots:
        obj.clear()
        obj.update(snapshot)
        globals()[name] = obj

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 272–273\n"
        "Run 272 failed because factual_eval194 requires a complete tensor "
        "dictionary, not the ten-tensor submission backup; globals restored "
        "and no saved weights changed. Run 273 supplied all 181 tensors. "
        f"CPU control {r273_control_nll}; candidate results {r273_results}. "
        "NE symmetry candidates remain unsaved pending selection and GPU tests.\n"
    )
```