Both drugs have measurable next-hour effects in the records, but the original rollouts barely respond to dose changes. The earlier posterior-action ablation did not explain that gap. I’ll inspect the local update helper and measure how strongly the action encoder represents dose; this will distinguish weak dose encoding from weak downstream propagation.

```python
import inspect, dis, io

print("--- Update helper source or compact bytecode ---")
for fname in ["context_from_saved", "wm_step_saved"]:
    fun = globals()[fname]
    print("\nFUNCTION", fname, "defaults", fun.__defaults__)
    try:
        print(inspect.getsource(fun))
    except (OSError, TypeError):
        instructions = list(dis.get_instructions(fun))
        text = "; ".join(
            f"{i.opname} {i.argrepr}" for i in instructions
            if i.opname not in {"RESUME", "PRECALL", "CACHE", "NOP", "PUSH_NULL"}
        )
        print(text[:6500])

print("\n--- Encoder helper conventions ---")
for fname in ["action_atom_exact", "set_encode_single_local", "wm_set"]:
    fun = globals()[fname]
    print(fname, "defaults", fun.__defaults__, "constants", repr(fun.__code__.co_consts)[:500])

print("\n--- Dose encoding over observed and extrapolated rates ---")
dose_encoder_probe = {}
for drug, grid, ratio in [
    ("insulin", np.array([0., .5, 1., 2., 4., 8., 12., 16., 24.]), 1.),
    ("norepinephrine", np.array([0., .01, .03, .05, .1, .2, .4, .8, 1., 2.]), 4.8),
]:
    atoms = action_atom_exact(drug, grid, grid * ratio)
    enc = set_encode_single_local(atoms, "encode_action")
    enc = enc.detach().cpu().numpy() if isinstance(enc, torch.Tensor) else np.asarray(enc)
    dose_encoder_probe[drug] = (grid, enc)
    print(drug, "shape", enc.shape,
          "distance from lowest rate", np.round(np.linalg.norm(enc - enc[0], axis=-1), 3))
print("\n--- Nuisance model types ---")
for name, model in nuisance_models.items():
    print(name, type(model).__name__)
```
