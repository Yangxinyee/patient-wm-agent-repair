The controller’s factual variation is real, so I won’t install it unchanged. I’ll first test a shorter correction path: change the next-reading response to dose and let the model’s existing reading feedback carry it forward. This simpler route is worth keeping only if rollouts improve without violating the factual NLL limit.

```python
print("=== WIRE IDENTITIES ===")
bin138 = w63("core.b_in.weight").numpy()
print("c_out @ diag(delta) @ T125 vs R118:",
      np.linalg.norm(cout137 @ (delta137[:,None]*T125)-R118))
print("SP125 vs c_out @ diag(delta) @ b_in @ QE125:",
      np.linalg.norm(SP125-cout137 @ (delta137[:,None]*(bin138@QE125))))
print("T125 vs inv(c_out)R/delta:",
      np.linalg.norm(T125-np.linalg.solve(cout137,R118)/delta137[:,None]))
print("T125 vs inv(c_out)QE/delta:",
      np.linalg.norm(T125-np.linalg.solve(cout137,QE125)/delta137[:,None]))
print("private decoder extraction response Jselected @ Vsignal:",
      np.round(Jselected118 @ Vsignal118,4))
print("private readout factual mean/SD:",
      np.round(xf118.mean(axis=0)@Vsignal118,4),
      np.round((xf118@Vsignal118).std(axis=0),4))
for k in ["route122", "transmit122", "leak122", "sub125", "constraints125"]:
    if k in globals():
        x = globals()[k]
        print(k, tuple(x.shape) if hasattr(x,"shape") else str(x)[:150])
        if hasattr(x,"size") and x.size <= 20:
            print(x)

print("\n=== INPUT INTERFACES ===")
compact_bytecode136("bundle_inputs73")
ids138 = np.asarray(valid_list103[:4], dtype=int)
inputs138 = bundle_inputs74(ids138)
print("sample ids", ids138)
print("bundle inputs:",
      [(type(x).__name__, tuple(x.shape) if hasattr(x,"shape") else str(x)[:100])
       for x in inputs138])
print("Retained decoder inputs:")
for k in list(globals()):
    if re.search(r"(67|73|81|96|102)$",k) and isinstance(globals()[k],(np.ndarray,torch.Tensor,pd.DataFrame)):
        x = globals()[k]
        print(k, tuple(x.shape))

print("\n=== HEAD ORIGINAL WEIGHTS ===")
for name in ["attr_heads.value_bin.0.weight","attr_heads.value_bin.0.bias",
             "attr_heads.value_bin.2.weight","attr_heads.value_bin.2.bias"]:
    x = load_original(name)
    print(name, tuple(x.shape), "norm", float(np.linalg.norm(x)))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 138 / simplify before persistence\n"
            "P125 finite-difference error ~.01 but factual SD .62/.90/1.64/.41; "
            "not safe to install unchanged. First test an immediate private-state dose "
            "wire without controller, relying on existing reading feedback. "
            "Controller remains an unvalidated alternative. Reconstruction-only edits "
            "still the only saved changes; forward 5/50.\n")
```