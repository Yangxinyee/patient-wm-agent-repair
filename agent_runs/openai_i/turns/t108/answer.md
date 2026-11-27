The END grid failed: its errors are far too large to justify an edit. I’ll discard that reconstruction and check whether END atoms enter a different encoder—or are omitted—using the preserved helpers. Agreement on both simultaneous start/END bundles and END-only bundles is the required check.

```python
import dis

print("PRESERVED ENCODER HELPERS")
for _name100 in ("complete_atom57", "single_set57", "empty_set58",
                 "post_mean59", "latent_categories", "repair_mlp_act"):
    _fn100 = globals()[_name100]
    print(_name100, inspect.signature(_fn100),
          "constants:", repr(_fn100.__code__.co_consts)[:450],
          "dependencies:", _fn100.__code__.co_names)

print("\nCOMPLETE ATOM IMPLEMENTATION")
print(" | ".join(f"{i.opname} {i.argrepr}" for i in
                 dis.get_instructions(complete_atom57))[:2700])

print("\nENCODER TENSOR NAMES")
print([(n, SHAPES[n]) for n in NAMES
       if n.startswith(("encode.", "encode_action."))])

print("\nIGNORE-END DIAGNOSTIC")
_has_start100 = np.array([len(c[1]) > 0 for c in _probe_cases99])
for _empty_kind100 in ("encoded", "zero"):
    _us100 = []
    for _case100 in _probe_cases99:
        if _case100[1]:
            _us100.append(latent_action_encode(
                [latent_start_atom(r) for r in _case100[1]]))
        else:
            _us100.append(u_all65[0] if _empty_kind100 == "encoded"
                          else torch.zeros(192))
    _us100 = torch.stack(_us100)
    _hidden100 = latent_hidden(_probe_ctx99, _us100)
    _means100 = repair_linear(_hidden100, "z_post.2")[:, :32]
    _error100 = torch.sqrt(torch.mean(
        (_means100 - _probe_target99)**2, dim=1)).numpy()
    print(_empty_kind100,
          "start+END RMS", float(_error100[_has_start100].mean()),
          "END-only RMS", float(_error100[~_has_start100].mean())
          if (~_has_start100).any() else None)

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 100 — rejected END reconstruction\n"
             "- Python 99 END-in-action grid failed (mean posterior RMS about 0.097); "
             "do not use those features for editing.\n"
             "- Testing omission/routing of END atoms and recovering earlier validated "
             "encoder helper contracts. No submitted weights changed.\n")
```