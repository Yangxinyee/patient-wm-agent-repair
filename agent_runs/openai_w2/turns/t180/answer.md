The corrected analyses support insulin lowering glucose by roughly 1.6–2.0 per rate unit and norepinephrine raising MAP by roughly 3.5–3.9. Persistence is less certain, yet strongly affects the 12-hour targets. I’ll inspect and extend the statistical persistence fit; stable estimates would support the accumulated effects, while continued drift would weaken them.

```python
print("=== PERSISTENCE SENSITIVITY NOTES ===")
a156 = notes154.find("## Run 12:")
b156 = notes154.find("## Restart, run 13", a156)
print(notes154[a156:b156])

print("\n=== CONVERGED LATENT-FIT NOTES ===")
a156b = notes154.find("## Run 39:")
b156b = notes154.find("## Forward 7", a156b)
print(notes154[a156b:b156b][:2500])

print("\n=== HELPER CONTRACTS ===")
for name156 in ("sensor_iv_persistence", "fit_latent_record_model"):
    fn156 = globals()[name156]
    print(name156, inspect.signature(fn156))
    print("Local variables:", fn156.__code__.co_varnames)
    print("Referenced names:", fn156.__code__.co_names)
    print("Docstring:", fn156.__doc__)

print("\n=== STORED LATENT PARAMETERS ===")
for name156, obj156 in list(globals().items()):
    if isinstance(obj156, dict):
        if "rho" in obj156:
            print(name156, obj156)
        elif any(isinstance(v, dict) and "rho" in v for v in obj156.values()):
            print(name156, obj156)
    elif isinstance(obj156, pd.DataFrame) and any(
        s in name156.lower() for s in ("persist", "latent", "target", "dml")
    ):
        print(name156, obj156.to_string(index=False)[:1200])

print("\n=== PANEL ALIGNMENT ===")
print("Index names:", panel.index.names)
print(panel.head(5).to_string())
print("Hours:", panel.index.get_level_values("hour").unique().tolist())

print("\n=== CONSTRAINED DECODER ENTRY ===")
entry156 = constraint_candidates147[(0.3, 0.0001)]
print(type(entry156).__name__)
if isinstance(entry156, dict):
    for key156, value156 in entry156.items():
        if hasattr(value156, "shape"):
            print(key156, type(value156).__name__, value156.shape)
        else:
            print(key156, str(value156)[:700])
else:
    print(getattr(entry156, "shape", None))
```
