The retained analyses include controlled impulse responses and original states for 100 training stays, so we can test an edit without using unseen evaluation stays. I’ll inspect the posterior helpers and bundle metadata next. The key test is whether they distinguish a changed dose from the dose normally selected for that condition; otherwise an edit could merely amplify confounding.

```python
import inspect
import dis

print("=== CONFIG / IMPULSE ESTIMATES ===")
print(VOCAB["config"])
print(irf_table.to_string(index=False))
print("\n=== HELPER SIGNATURES ===")
for n in ["repair_atom_base", "latent_context", "latent_start_atom",
          "latent_categories", "latent_action_encode", "latent_encode_batch",
          "latent_hidden", "latent_post_mean", "head_exact"]:
    fn = globals()[n]
    print(n, inspect.signature(fn))
    print(" names:", fn.__code__.co_names,
          " constants:", [c for c in fn.__code__.co_consts
                         if isinstance(c, (str, int, float, tuple))][:25])

print("\n=== BUNDLE METADATA OBJECTS ===")
for n, v in list(globals().items()):
    if n.startswith("_") or n in {"repair_weights", "repair_lookup",
                                "latent_starts", "latent_ends"}:
        continue
    if any(s in n.lower() for s in ("state", "latent", "action", "clean", "repair_ids")):
        if isinstance(v, (list, tuple)):
            print(n, type(v).__name__, len(v), repr(v[:2])[:230])
        elif isinstance(v, np.ndarray) and v.ndim == 1:
            print(n, v.shape, repr(v[:5])[:180])

print("\n=== REPRESENTATIVE INDEX ENTRIES ===")
print("repair_lookup:", list(repair_lookup.items())[:2])
print("latent_starts:", list(latent_starts.items())[:2])
print("latent_ends:", list(latent_ends.items())[:2])

print("\n=== THREE IMPLEMENTATIONS ===")
for n in ["latent_context", "repair_atom_base", "latent_action_encode"]:
    print("\n", n)
    try:
        print(inspect.getsource(globals()[n]))
    except (OSError, TypeError):
        dis.dis(globals()[n])
```
