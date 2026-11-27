The candidate uses four low-variance action directions to carry dose and drug-presence information. It has not yet been tested in the full model. I’ll locate the observation-encoding discrepancy and verify the posterior against exported states; that agreement is necessary to turn a dose residual into a meaningful state correction.

```python
print("=== RECONSTRUCTION CACHE ===")
for name in sorted(list(globals())):
    if name.startswith("r36_"):
        obj = globals()[name]
        if not callable(obj):
            print(name, summary69(obj))

print("\n=== HELPER IMPLEMENTATION DETAILS ===")
for name in ("r36_atom", "r36_set_encode", "rr58_observation_encoding",
             "rr50_action_parts", "mlp", "r33_mlp", "r33_act"):
    obj = globals().get(name)
    if callable(obj):
        print(name, "names", obj.__code__.co_names,
              "constants", repr(obj.__code__.co_consts)[:900])
        try:
            src70 = inspect.getsource(obj)
            if name in ("rr58_observation_encoding", "r36_atom"):
                print(src70[:2600])
        except (OSError, TypeError):
            pass

print("\n=== FIRST CACHE IDENTITIES ===")
print("r36_indices", np.asarray(r36_indices)[:12])
print("state identities", np.asarray(r33_meta)[:12])
print("selected state identities", np.asarray(r33_meta)[np.asarray(r36_indices)[:12]])
print("rr58 sample rows", [(int(rows[0].stay_id), float(rows[0].hour), len(rows))
                            for rows in rr58_mrows[:4]])
if "r36_owners" in globals():
    for i in np.asarray(rr58_mi)[:4]:
        ix70 = np.flatnonzero(np.asarray(r36_owners) == i)
        print("cache row", int(i), "identity", r33_meta[int(r36_indices[i])],
              "cached atom count", len(ix70), "atom indexes", ix70[:8])

print("\n=== ACTION TRACE DETAILS ===")
print("scales", rr50_scales, "epsilon", rr50_epsilon)
print("P orthogonality error",
      np.max(np.abs(rr58_P.T @ rr58_P - np.eye(4))))
print("reading direction norms",
      {k: float(np.linalg.norm(v)) for k, v in rr58_reading_dirs.items()})
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 70\n"
            "Trace candidate is four low-variance action-output directions, not a validated edit. "
            "Inspecting exact cached atom ownership and helper activations to resolve reconstruction discrepancy. "
            "No tensors saved. Counts: 70 Python runs, 6 forward passes.\n")
```