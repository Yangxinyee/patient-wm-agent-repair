The gate fix stays within the factual limit, but insulin doubling now overshoots and halving remains too weak. Removing the no-drug leak was not enough. I’ll inspect the dose decoder and hourly rollout shifts to distinguish an asymmetric dose signal from nonlinear feedback; accurate paired dose contrasts would refute the decoder explanation.

```python
print("=== EXACT HELPERS ===")
r210_budget = 5400
for name in ["latent206", "prelude155"]:
    fn = globals().get(name)
    if callable(fn):
        try:
            src = inspect.getsource(fn)
        except Exception as exc:
            src = "Source unavailable: " + str(exc)
        print(name, src[:r210_budget])
        r210_budget -= min(len(src), r210_budget)

print("\n=== FOCUSED RETAINED VARIABLES ===")
r210_names = [
    k for k in list(globals())
    if not k.startswith("_") and (
        k.endswith(("199", "203", "206", "207"))
        or (callable(globals()[k]) and any(
            s in k.lower() for s in ["latent", "replay", "sensor", "transport", "response"]
        ))
    )
]
for k in sorted(r210_names):
    v = globals()[k]
    if callable(v):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k, "callable")
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v))
    elif isinstance(v, (int, float, str, bool)):
        print(k, repr(v)[:120])

print("\n=== ROLLOUT FILE ===")
r210_dose = pd.read_csv("/work/fwd_14/dose_edit_edited.csv")
print("columns:", list(r210_dose.columns))
print(r210_dose.head(4).to_string(index=False))
if {"edit", "hour", "shift"}.issubset(r210_dose.columns):
    r210_rows = r210_dose
    if "reached" in r210_rows:
        r210_rows = r210_rows[r210_rows["reached"].astype(bool)]
    print(r210_rows.groupby(["edit", "hour"])["shift"].mean().unstack(0).round(3).to_string())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 210; forwards 14. Gate-fixed restricted candidate "
        "hash b0e94cf0b5: factual NLL 1.64302 vs 1.63982 (+.00320). "
        "Rollout shifts insulin x2 -19.175, half +1.421; NE x2 +1.701, "
        "half -1.245, versus record-derived references -12.9555,+6.4778,"
        "+4.2797,-2.1398. No-action leak fix alone does not resolve "
        "insulin asymmetry. Inspecting exact decoder and hourly dynamics. "
        "Candidate remains installed; checkpoint193 retained.\n"
    )
```