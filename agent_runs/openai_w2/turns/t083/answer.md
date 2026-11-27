The saved states include both the recurrent slots and posterior latent means, which should let us test a correction on ordinary drug bundles without reconstructing the whole stay. I’ll check their event layout and encoder conventions next. The route is usable only if those bundles reproduce accurately and retain a measurable dose signal.

```python
import inspect

print("=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text()[:5000])

print("\n=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
for key in ("config", "codes", "normalization", "normalisation", "units", "categorical"):
    if key in VOCAB:
        print(key, str(VOCAB[key])[:1800])

print("\n=== RECONSTRUCTION HELPER INTERFACES ===")
for name in (
    "rlinear", "rnorm", "rmlp", "rmlp_act", "action_encode_atoms",
    "action_encode_variant", "embed_record_atom", "encode_atom_set",
    "local_value_head"
):
    func = globals()[name]
    print(name, inspect.signature(func))
    print("  names:", func.__code__.co_names)
    print("  constants:", str(func.__code__.co_consts)[:700])

print("\n=== REFERENCE STATE METADATA ===")
print(state_meta.head(8).to_string(index=False))
print("action bundle counts:",
      int(np.sum(S_reference["has_insulin"])),
      int(np.sum(S_reference["has_norepinephrine"])))
print("atom-count distribution:", np.unique(S_reference["n_atoms"], return_counts=True))

print("\n=== ONE STAY'S FIRST EVENTS ===")
print(reference_records_repair.head(24).to_string(index=False))
```
