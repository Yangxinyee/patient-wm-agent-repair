The norepinephrine columns are only approximately opposite; they retain other learned components. I’ll screen shared-gate edits both with those columns preserved and with their common component removed. Factual fit will determine whether either construction is worth a rollout test; neither is yet a saved repair.

```python
def symmetric_ne272(gain, exact_pair=False, gate_shift=0.0):
    c = symmetric_ne271(gain, gate_shift)
    if exact_pair:
        o = a262(c["z_post.2.weight"]).copy()
        direction = (o[:, 156] - o[:, 89]) / 2
        o[:, 156], o[:, 89] = direction, -direction
        c["z_post.2.weight"] = like258(o, c["z_post.2.weight"])
    return c

r272_snapshots = [
    (name, obj, dict(obj))
    for name, obj in list(globals().items())
    if isinstance(obj, dict) and "z_post.0.weight" in obj
]
r272_results, r272_candidates = {}, {}
try:
    _, losses, _ = factual_eval194(backup269)
    print("CURRENT CPU CONTROL", float(np.mean(losses)),
          "expected", 1.6392319202423096)
    for g, exact in [(0.8, False), (1.0, False), (0.8, True), (1.0, True)]:
        c = symmetric_ne272(g, exact)
        _, losses, _ = factual_eval194(c)
        nll = float(np.mean(losses))
        r272_results[(g, exact)] = {
            "nll": nll, "delta": nll - 1.6355364322662354
        }
        r272_candidates[(g, exact)] = c
    print("UNSAVED NE SCREEN", r272_results)
finally:
    for name, obj, snapshot in r272_snapshots:
        obj.clear()
        obj.update(snapshot)
        globals()[name] = obj

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 272\n"
        "NE pair output columns include a nonzero common/native component. "
        "Screened shared-gate symmetry with native columns retained and with "
        "their common component removed, at gains .8 and 1.0. "
        f"CPU results {r272_results}. No saved weights changed; model-weight "
        "dictionary globals restored. GPU validation is still required.\n"
    )
```