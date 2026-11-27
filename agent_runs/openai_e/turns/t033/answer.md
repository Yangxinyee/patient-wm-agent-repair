The retained results rule out a simple reading-head correction: those features recover policy-associated doses but almost none of the dose innovations. I’ll inspect the posterior pathway instead. A useful correction must detect dose changes at drug bundles while remaining quiet on factual trajectories; failure to recover held-out dose innovations would refute this approach.

```python
print("=== POSTERIOR / ACTION TENSORS ===")
for _n30 in NAMES:
    if any(_s30 in _n30 for _s30 in
           ("z_post", "z_prior", "z_in", "z_head", "encode_action")):
        print(_n30, SHAPES[_n30])

print("=== SAVED STATE FILE ===")
_s30 = np.load("/work/fwd_3/states_original.npz")
print([(k, _s30[k].shape, str(_s30[k].dtype)) for k in _s30.files])

print("=== RELEVANT DICTIONARIES ===")
for _k30, _v30 in sorted(list(globals().items())):
    if not _k30.startswith("_") and isinstance(_v30, dict):
        print(_k30, len(_v30), list(_v30)[:5])

print("=== REMAINING ARRAY NAMES ===")
for _k30, _v30 in sorted(list(globals().items())):
    if _k30.startswith("_") or _k30 < "policy_":
        continue
    if isinstance(_v30, (np.ndarray, torch.Tensor)):
        print(_k30, tuple(_v30.shape))
    elif isinstance(_v30, pd.DataFrame):
        print(_k30, "DataFrame", _v30.shape)

print("=== RECONSTRUCTION DEFINITIONS ===")
for _fn30 in (recon_ctx, recon_record_atom, recon_sets, recon_single_set):
    print("\nFUNCTION:", _fn30.__name__)
    try:
        print(inspect.getsource(_fn30)[:1800])
    except Exception:
        print("Source unavailable; globals:", _fn30.__code__.co_names)
        print("Constants:", repr(_fn30.__code__.co_consts)[:900])

with open("/work/notes.md", "a") as _f30:
    _f30.write(
        "\n## Run 30: targeted posterior recovery\n"
        "Reading-head dose-innovation probes failed. Inspecting action-specific "
        "posterior features and exact saved-state format to test a factually "
        "centred latent correction. No edits.\n"
        "Python runs: 30; forward passes: 3.\n"
    )
```