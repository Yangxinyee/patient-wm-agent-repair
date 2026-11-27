The reported “start” group includes measurements, so its error does not yet isolate drug-start accuracy. I’ll separate bundles by event and inspect the atom defaults. If a few categorical defaults explain the mismatch, we can repair the local reconstruction without changing any model weights.

```python
import dis
print("=== VOCAB KINDS ===")
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])
print("=== ATOM BUILDER: CATEGORICAL SECTION ===")
instructions = list(dis.get_instructions(make_atom))
for ins in instructions[-115:]:
    print(ins.offset, ins.opname, ins.argrepr)
print("=== CATEGORICAL TABLES ===")
for n in NAMES:
    if n.startswith(("embed.categorical", "embed.kind", "embed.ekind")):
        print(n, SHAPES[n])

print("=== ERROR FOR ISOLATED EVENTS ===")
pos_by_idx = {int(idx): p for p, idx in enumerate(valid_idx)}
isolated_by_event = {}
for idx in valid_idx:
    info = bundle_atom_info[idx]
    if len(info) == 1:
        rec, ending = info[0]
        label0 = rec.event + (" END" if ending else "")
        isolated_by_event.setdefault(label0, []).append(int(idx))
for label0, idxs in isolated_by_event.items():
    positions0 = np.array([pos_by_idx[idx] for idx in idxs])
    err0 = posterior_error[positions0]
    print(label0, len(idxs), "mean", float(err0.mean()),
          "median", float(np.median(err0)), "max", float(err0.max()))
```