The retained design uses four low-variance state directions, but the failed edit wrote full-rate signals into them. That may have pushed the recurrent computation far outside its usual range. I’ll check the write scale and controller equations; smaller, well-isolated signals would support this explanation, while large errors even without injection would implicate the controller itself.

```python
print("=== TRANSITION IMPLEMENTATION ===")
skeleton149(transition64, limit=4200)
print("\n=== PRELUDE FEATURE IMPLEMENTATION ===")
skeleton149(prelude_features102, limit=2000)
print("\n=== PROJECTED TIME FLOW ===")
skeleton149(projected_flow153, limit=1200)

print("\n=== RECENT DESIGN VARIABLES ===")
for n in list(globals()):
    if n.endswith(("152", "153", "159", "160")):
        obj = globals()[n]
        if hasattr(obj, "shape"):
            print(n, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(n, "dict", list(obj.keys())[:10])
        elif isinstance(obj, (int, float, str, list, tuple)):
            print(n, repr(obj)[:250])

print("\n=== CONTROLLER COMPARISONS ===")
C164 = np.asarray(load_original("core.c_out.weight"))
B164 = np.asarray(load_original("core.b_in.weight"))
for penalty, ctrl in controllers159.items():
    a = np.asarray(ctrl)
    print("penalty", penalty, "norm", round(float(np.linalg.norm(a)), 4),
          "P-write gain", np.round(a.T @ Uexact145, 3).tolist())
print("C original/modified norms", float(np.linalg.norm(C164)),
      float(np.linalg.norm(Ccontroller145-C164)))
print("wire increment norm", float(np.linalg.norm(np.asarray(wire_out152)-np.asarray(Wrepair125))))
print("R private orthogonality", np.round(R118.T @ R118, 4))
print("\n=== ORIGINAL CACHE ENTRY CHECK ===")
for n in ["core.b_in.weight", "core.c_out.weight", "core.prelude.0.attn.kv.weight",
          "core.prelude.0.attn.out.weight"]:
    a = w63_cache.get(n)
    print(n, None if a is None else tuple(a.shape),
          None if a is None else float(np.linalg.norm(np.asarray(a)-np.asarray(load_original(n)))))
print("\n=== KERNELS ===")
print("kernels19", kernels19)
print("poles153", poles153)
```