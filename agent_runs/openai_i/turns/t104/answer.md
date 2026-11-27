The original model’s rollout effects are much smaller than the lagged estimates from the records. A value-head-only edit looks weak: its features barely decode dose innovations, especially for norepinephrine. I’ll instead test a posterior edit that carries dose innovations into the shared recurrent state, starting with the preserved action-feature alignment.

```python
print("ACTION/STATE ALIGNMENT OBJECTS")
for _k, _v in list(globals().items()):
    if (_k.startswith(("latent_", "action_", "clean_", "rates"))
        or any(s in _k.lower() for s in ("indices50", "index50", "residual", "gap_prefix"))):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_k, tuple(_v.shape), str(_v.dtype),
                  "first:", str(_v.reshape(-1)[:5])[:100])
        elif isinstance(_v, (list, tuple)):
            print(_k, type(_v).__name__, len(_v), repr(_v[:2])[:170])
        elif isinstance(_v, dict):
            print(_k, "dict", list(_v)[:10])
        elif not callable(_v) and not isinstance(_v, pd.DataFrame):
            print(_k, repr(_v)[:170])

print("\nVOCABULARY INPUT FIELDS")
for _key96 in ("config", "codes", "event_kinds", "atom_kinds"):
    print(_key96, repr(VOCAB[_key96])[:1500])
print("numeric spec sample", repr(VOCAB["numeric"])[:1800])
print("categorical tensors",
      [(n, SHAPES[n]) for n in NAMES if "embed.categorical" in n])

print("\nPRESERVED DOSE-RESPONSE DIRECTIONS")
for _drug96, _v96 in dose_directions50.items():
    print(_drug96, "norm", float(np.linalg.norm(_v96)),
          "components", np.round(_v96[:10], 5).tolist())

print("\nKEY IMPLEMENTATIONS")
for _name96 in ("latent_start_atom", "latent_hidden", "latent_context", "action_vector50"):
    try:
        _src96 = inspect.getsource(globals()[_name96])
        print(_name96, _src96[:2200])
    except (OSError, TypeError):
        _fn96 = globals()[_name96]
        print(_name96, "signature", inspect.signature(_fn96),
              "constants", repr(_fn96.__code__.co_consts)[:500],
              "locals", _fn96.__code__.co_varnames)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 96 — dose-specific posterior plan\n"
             "- Record-derived lag targets substantially exceed original rollout effects.\n"
             "- Head-hidden dose-innovation decoding is near zero; prefer a shared-state "
             "posterior correction, centered on factual policy.\n"
             "- Recovering action-feature alignment for a low-rank statistical edit; no edits yet.\n")
```