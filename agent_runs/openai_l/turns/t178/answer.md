The fitted controller has much larger coefficients than the original private-state mapping, so the problem may not be the dose write alone. I’ll isolate the controller with the original input weights and no dose injection. If factual fit still deteriorates, the persistence controller—not the wire—is responsible.

```python
print("=== TRANSITION IMPLEMENTATION ===")
print(skeleton149(transition64, limit=5000))
print("\n=== PRELUDE FEATURE IMPLEMENTATION ===")
print(skeleton149(prelude_features102, limit=2200))
print("\n=== PROJECTED TIME FLOW ===")
print(skeleton149(projected_flow153, limit=1500))
print("\n=== BUNDLE INPUT INTERFACE ===")
print(skeleton149(bundle_inputs74, limit=2200))

print("\n=== STAGED ROUTE SCALES ===")
for n in ["P160", "U160", "D160", "B160", "C160", "delta160", "systems152",
          "wire_slopes153", "chosen125", "rate_state81", "state_rows118"]:
    obj = globals().get(n)
    if isinstance(obj, dict):
        print(n, {k: (np.round(np.asarray(v), 5).tolist() if hasattr(v, "shape") and np.asarray(v).size < 40
                       else repr(v)[:400]) for k, v in obj.items()})
    elif hasattr(obj, "shape"):
        a = np.asarray(obj)
        print(n, a.shape, "norm", float(np.linalg.norm(a)),
              "range", [float(a.min()), float(a.max())],
              "sample", a.flatten()[:8].tolist())
    else:
        print(n, repr(obj)[:600])
print("B increment norm", float(np.linalg.norm(np.asarray(B160)-B164)))
print("C increment norm", float(np.linalg.norm(np.asarray(C160)-C164)))

drop_edit("core.b_in.weight")
save_edit("core.c_out.weight", torch.as_tensor(C160, dtype=torch.float32))
save_edit("core.prelude.0.attn.out.weight", torch.as_tensor(Wrepair125, dtype=torch.float32))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 165 / controller isolation\n"
            "Staged c_out=C160 alone alongside neutral attention reconstruction; "
            "b_in original, no dose wire or readout. This isolates recurrent-controller drift.\n")
print("CONTROLLER-ONLY EDITS", edited())
```