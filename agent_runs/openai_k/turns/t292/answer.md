The recovered notes support substantial effects from the records: roughly −2 glucose units per insulin rate-unit initially, and +3 MAP units per norepinephrine rate-unit, with effects persisting across hours. The output was truncated before the helper inventory. I’ll retrieve only that inventory now so the next edit uses the validated replay rather than a reconstructed approximation.

```python
def rr_inventory():
    env = list(globals().items())
    lines = ["=== REPLAY / FIT FUNCTIONS ==="]
    for name, obj in env:
        if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__" and not name.startswith("rr"):
            try:
                lines.append(name + "(" + ", ".join(inspect.signature(obj).parameters) + ")")
            except Exception:
                lines.append(name + "(?)")

    lines.append("\n=== RECENT ARRAYS AND DATA ===")
    objects = []
    for name, obj in env:
        if name.startswith("_") or name.startswith("rr"):
            continue
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            objects.append(f"{name}: {type(obj).__name__} {tuple(obj.shape)}")
        elif isinstance(obj, dict):
            objects.append(f"{name}: dict keys={list(obj.keys())[:6]}")
        elif hasattr(obj, "columns") and hasattr(obj, "shape"):
            objects.append(f"{name}: DataFrame {tuple(obj.shape)} columns={list(obj.columns)[:6]}")
    lines.extend(objects[-110:])
    print("\n".join(lines)[:9000])

    print("\n=== SAVED CANDIDATE KEYS ===")
    for fn in ["pre_recurrent_candidate.npz", "recurrent_half_backup.npz",
               "recurrent_glucose_valid.npz"]:
        p = pathlib.Path("/work") / fn
        if p.exists():
            with np.load(p) as z:
                print(fn, [(k, z[k].shape) for k in z.files])
    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 259: compact interface inventory\n"
                "- Run 258 output was truncated before the helper inventory.\n"
                "- Record-based immediate and dynamic effects recovered; no medical assumptions used.\n"
                "- Recovering compact helper and candidate metadata. No weights changed.\n"
                "- Python runs: 259; forward passes: 21.\n")

rr_inventory()
```