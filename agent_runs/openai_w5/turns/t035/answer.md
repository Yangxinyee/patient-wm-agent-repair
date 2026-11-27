The saved objects include causal estimates and original states, but the previous head-only dose decoder did not generalize. I’ll inspect the state format and input conventions, then reconstruct one recurrent update locally. Matching saved GPU states will be the check that makes subsequent weight-edit experiments trustworthy.

```python
import torch, inspect
torch.set_num_threads(8)
torch.set_grad_enabled(False)

print("--- Record schema ---")
print(Path("/data/README.md").read_text()[:5000])

print("\n--- Input conventions ---")
for key in ("atom_kinds", "event_kinds", "numeric_note"):
    print(key, VOCAB.get(key))
print("normalisation:", norm_meta)
print("head conventions:", head_spec, head_layout_spec)

print("\n--- State files ---")
for run in (3, 4, 5):
    path = Path(f"/work/fwd_{run}/states_original.npz")
    if path.exists():
        data_probe = np.load(path)
        print(str(path), [(k, data_probe[k].shape, str(data_probe[k].dtype))
                          for k in data_probe.files])
        for k in data_probe.files:
            if data_probe[k].ndim == 1:
                print(k, data_probe[k][:8])
        data_probe.close()

print("\n--- Cached arrays ---")
for cache_name in ("cacheA", "cacheTrain", "cacheTrainB"):
    cc_probe = globals()[cache_name]
    print(cache_name, [(k, getattr(v, "shape", None)) for k, v in cc_probe.items()])
    print(cc_probe["df"].head(3).to_string(index=False))

print("\n--- Helper definitions ---")
for helper_name in ("lin0", "norm0", "mlp0", "reading_cache"):
    helper_probe = globals()[helper_name]
    try:
        print(inspect.getsource(helper_probe)[:1600])
    except Exception:
        print(helper_name, "constants:", helper_probe.__code__.co_consts,
              "names:", helper_probe.__code__.co_names)
print("W0 tensor type:", type(W0["embed.numeric.weight"]))
print("Tuning A ids:", target_ids_A[:8])
```
