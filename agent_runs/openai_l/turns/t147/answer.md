The retained fits include dose-contrast regressions across several multipliers and a four-state controller. They are useful components, not yet a validated repair. I’ll check their algebra and decoder interface next; success requires correct state transmission, not merely a good regression fit.

```python
def compact_bytecode136(name):
    f = globals()[name]
    print("\n", name, inspect.signature(f))
    print("constants:", f.__code__.co_consts)
    print("globals:", f.__code__.co_names)
    lines = []
    current = []
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None and current:
            lines.append(" ".join(current))
            current = []
        op = {"LOAD_GLOBAL":"G", "LOAD_FAST":"F", "LOAD_CONST":"C",
              "LOAD_ATTR":"A", "STORE_FAST":"S", "CALL":"CALL",
              "BINARY_SUBSCR":"[]", "BINARY_OP":"OP", "BUILD_SLICE":"SL",
              "BUILD_TUPLE":"TUP", "BUILD_LIST":"LIST"}.get(ins.opname, ins.opname)
        if ins.opname not in {"RESUME", "CACHE", "RETURN_VALUE"}:
            current.append(op + (":" + ins.argrepr if ins.argrepr else ""))
    if current:
        lines.append(" ".join(current))
    print("\n".join(lines))

for name in ["transition64", "context64", "head_inputs96", "attention_features116"]:
    compact_bytecode136(name)

print("\n=== CONTROLLER ALGEBRA ===")
for p, b in controllers125.items():
    print("P125 vs penalty", p, "RMS", float(np.sqrt(np.mean((P125-b)**2))))
print("controller residual RMS:", np.sqrt(np.mean((Xstate125 @ P125-Ystate125)**2, axis=0)))
print("controller factual RMS:", np.sqrt(np.mean((Ef124 @ P125)**2, axis=0)))
print("R118 Gram:", np.round(R118.T @ R118, 4))
print("SP125 to private state:", np.round(R118.T @ SP125, 4))
print("T125 vs invroute124 RMS:", float(np.sqrt(np.mean((T125-invroute124)**2))))
print("small variables:",
      {k: globals()[k] for k in ["gamma124", "penalty125", "chosen125"]
       if k in globals()})
print("dose coefficients norm/intercept:", np.linalg.norm(Bdose125[:-1], axis=0), Bdose125[-1])
print("FULL RECORD KERNELS:", kernels19)

print("\n=== FORWARD/DECODER INTERFACES ===")
for name in ["decode67", "bundle_inputs73", "bundle_inputs74", "prepare103",
             "prelude_features102", "make_cf81"]:
    f = globals()[name]
    print(name, inspect.signature(f), "globals:", f.__code__.co_names,
          "constants:", str(f.__code__.co_consts)[:180])
print("Relevant retained array names:")
print(", ".join(k for k in globals()
                if re.search(r"(60|64|73|74|81|96|102|103)$", k)))
```