The helpers exist, but their source is unavailable and the verbose bytecode was truncated. I’ll extract only the relevant operations and compare the two drug-atom builders. This should tell us whether the discrepancy comes from defaults or binning, rather than the recurrent model.

```python
import dis, collections, sys

def compact_bytecode(func):
    groups = collections.OrderedDict()
    line = None
    for ins in dis.get_instructions(func):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname in ("RESUME", "CACHE", "NOP"):
            continue
        op = ins.opname.replace("LOAD_", "L_").replace("STORE_", "S_")
        groups.setdefault(line, []).append(op + (" " + ins.argrepr if ins.argrepr else ""))
    return "\n".join(f"{line}: " + "; ".join(ops) for line, ops in groups.items())

print("=== MAKE_ATOM ===")
print(compact_bytecode(make_atom))
print("\n=== ENCODE_ONE_SET ===")
print(compact_bytecode(encode_one_set))
print("\n=== ATOM_BASE CONSTANTS / NAMES ===")
print(atom_base.__code__.co_consts)
print(atom_base.__code__.co_names)
print("\n=== MLP ===")
print(compact_bytecode(mlp))

print("\n=== LATENT TENSORS ===")
for name in NAMES:
    if name.startswith(("z_", "gap.")):
        print(name, SHAPES[name])

print("\n=== ISOLATED SAMPLE FORMAT ===")
for ev in ("map_cuff", "insulin", "insulin END"):
    obj = isolated_by_event[ev]
    print(ev, type(obj).__name__, repr(obj[:1])[:450])

print("\n=== DRUG BUILDER COMPARISON ===")
for ev in ("insulin", "norepinephrine"):
    r = next(r for r in state_records.itertuples(index=False) if r.event == ev)
    am = torch.as_tensor(make_atom(r), dtype=torch.float32).reshape(1, 192)
    ab = torch.as_tensor(atom_base(ev, np.array([r.rate]), np.array([r.dose])), dtype=torch.float32).reshape(1, 192)
    print(ev, "atom MSE", float(torch.mean((am-ab)**2)),
          "record", r.rate, r.dose, r.rate_unit, r.dose_unit)

print("default-vs-kq0 MSE:", float(np.mean(
    (as_numpy(defaults) - as_numpy(W["embed.kq.weight"])[0])**2)))
print("history available:", "In" in globals(), "get_ipython" in globals())
```