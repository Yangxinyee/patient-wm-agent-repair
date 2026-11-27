The record-based estimates suggest an insulin effect of about −2 per unit/hour initially, persisting for several hours, and a norepinephrine effect of about +3 with a shorter tail. These estimates remain conditional on the identification assumptions. I’ll test whether posterior features can represent dose deviations from the usual policy, including altered rates. Accurate held-out decoding with little response on action-free bundles would support a low-rank edit.

```python
print("=== INPUT METADATA ===")
print("Vocabulary keys:", list(VOCAB))
print("Codes:", VOCAB["codes"])
print("H index:", H.index.names, list(H.index[:3]))
print("Dose residual shapes:", {k: np.asarray(v).shape for k, v in dose_residuals.items()})

print("\n=== ACTION HELPER SOURCE, IF AVAILABLE ===")
for _name in ["latent_start_atom", "latent_action_encode", "latent_hidden", "latent_context"]:
    _fn = globals()[_name]
    try:
        _src = inspect.getsource(_fn)
        print(_name, _src[:1800])
    except (OSError, TypeError):
        print(_name, "locals:", _fn.__code__.co_varnames,
              "constants:", [v for v in _fn.__code__.co_consts
                             if isinstance(v, (str, int, float, tuple))][:15])

probe_indices48 = np.array([
    i for i in range(len(latent_sid))
    if len(latent_starts.get(i, [])) == 1 and not latent_ends.get(i, [])
], dtype=int)
print("\nSingle-start, no-END bundles:", len(probe_indices48))
for _i in probe_indices48[:2]:
    print("Example:", _i, latent_sid[_i], latent_hour[_i],
          type(latent_starts[_i][0]).__name__, latent_starts[_i][0])

print("\n=== EARLIER IDENTIFICATION CHECKS ===")
print(rd_table.to_string(index=False))
print("\nCurrent edits:", edited())

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 48; forward 6 — correction proposal\n"
             "- Retrieved controlled impulse estimates: insulin initially -1.99 per rate unit, with a multi-hour tail; NE initially +3.29 with a shorter tail.\n"
             "- Considering a posterior-mean low-rank edit driven by decoded dose innovations, with synthetic rate changes used only to test representation/extrapolation.\n"
             "- Effects are estimated from records, not medical assumptions. No weights changed.\n")
```
